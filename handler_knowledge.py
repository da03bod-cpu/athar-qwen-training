"""Knowledge route dispatcher — runs on the same existing RunPod endpoint."""
from __future__ import annotations
import hashlib
import time
from knowledge.artifact_io import download, upload, MAX_SOURCE_BYTES
from knowledge.embedding import embedder
from knowledge.errors import KnowledgeError
from knowledge.extraction import extract_pages
from knowledge.indexing import make_pack, open_pack, serialize_pack, SCHEMA
from knowledge.retrieval import load_index, search_pack, clear_cache
from knowledge.security import same_scope, scope_from_resource, require_id


def _validate_scope_query(knowledge):
    if not isinstance(knowledge, dict):
        raise KnowledgeError("INVALID_REQUEST", "Missing knowledge object")
    return scope_from_resource(knowledge)


def _run_index(req, *, delete=False):
    resource = req.get("resource")
    scope, organization_id, project_id = scope_from_resource(resource)
    idx = req.get("index")
    if not isinstance(idx, dict):
        raise KnowledgeError("INVALID_REQUEST", "Index reference required")
    expected = idx.get("expected_generation")
    if type(expected) is not int or expected < 0:
        raise KnowledgeError("INVALID_REQUEST", "Invalid expected generation")
    embedding = embedder()
    old = None
    if expected > 0:
        ref = idx.get("previous_index_ref")
        if not ref:
            raise KnowledgeError("INDEX_NOT_READY", "Previous index missing")
        old, _ = load_index(ref, scope=scope, organization_id=organization_id, project_id=project_id,
                            generation=expected, embedding=embedding)
    elif idx.get("previous_index_ref"):
        raise KnowledgeError("INVALID_REQUEST", "Unexpected previous index for generation zero")
    pages = None
    if not delete and resource.get("source_ref"):
        raw = download(resource["source_ref"], max_bytes=MAX_SOURCE_BYTES)
        pages = extract_pages(raw, resource.get("mime_type"))
    new_pack = make_pack(previous=old, resource=resource, pages=pages,
                         embedding=embedding, generation=expected+1,
                         operation="delete" if delete else "upsert")
    if new_pack is None:
        return {"status": "INDEX_UNCHANGED", "type": req["type"], "run_id": req.get("run_id"),
                "resource_id": resource["resource_id"], "resource_revision": resource["revision"],
                "index_generation": expected}
    blob = serialize_pack(new_pack)
    digest = hashlib.sha256(blob).hexdigest()
    upload(idx.get("artifact_write_ref"), blob)
    # Backend must CAS publish; an upload alone is NOT an active generation.
    clear_cache()
    return {
        "status": "INDEX_ARTIFACT_UPLOADED", "type": req["type"],
        "run_id": req.get("run_id"), "scope": scope,
        "organization_id": organization_id, "project_id": project_id,
        "resource_id": resource["resource_id"], "resource_revision": resource["revision"],
        "expected_generation": expected, "proposed_generation": expected+1,
        "artifact_sha256": digest, "artifact_bytes": len(blob),
        "indexed_chunks": new_pack.manifest["chunk_count"],
        "embedding_model_id": embedding.id, "index_schema_version": SCHEMA,
    }


def run_search(req):
    knowledge = req.get("knowledge")
    scope, organization_id, project_id = _validate_scope_query(knowledge)
    generation = knowledge.get("index_generation")
    if type(generation) is not int or generation < 1:
        raise KnowledgeError("INVALID_REQUEST", "Invalid index_generation")
    encoding = embedder()
    pack, hit = load_index(knowledge.get("index_ref"), scope=scope, organization_id=organization_id,
                           project_id=project_id, generation=generation, embedding=encoding)
    rows = search_pack(pack, req.get("query", ""), max_chunks=knowledge.get("max_chunks", 8),
                       max_evidence_tokens=knowledge.get("max_evidence_tokens", 3500),
                       advisor_id=knowledge.get("advisor_id"), is_meta=knowledge.get("is_meta", False), encoding=encoding)
    return {"status": "COMPLETED", "type": "knowledge_search", "run_id": req.get("run_id"),
            "index_generation": generation, "index_cache_hit": hit, "hits": rows}


def dispatch(req):
    try:
        if req.get("preflight") is True:
            return {"status": "OK", "type": req.get("type"), "preflight": True,
                    "routes": ["knowledge_index", "knowledge_search", "knowledge_delete"],
                    "index_schema_version": SCHEMA,
                    "requires_backend_artifact_storage": True,
                    "indexing_on_inference": False}
        if req.get("type") == "knowledge_index":
            return _run_index(req)
        if req.get("type") == "knowledge_delete":
            return _run_index(req, delete=True)
        if req.get("type") == "knowledge_search":
            return run_search(req)
        raise KnowledgeError("INVALID_REQUEST", "Unknown knowledge operation")
    except KnowledgeError as exc:
        return {"status": "FAILED", "type": req.get("type"), "run_id": req.get("run_id"),
                "error_code": exc.code, "error": str(exc)}
    except Exception:
        # Avoid leaking signed refs, document payloads and tracebacks.
        return {"status": "FAILED", "type": req.get("type"), "run_id": req.get("run_id"),
                "error_code": "INTERNAL_ERROR", "error": "Knowledge operation failed"}
