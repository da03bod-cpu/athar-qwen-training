"""Immutable, scoped, incremental hybrid-index packs.

Pack contains only JSON, SQLite, NPY and optionally FAISS binary; never pickle.
Embedding text is performed only on changed resources. Generation publication
is deliberately the Backend's responsibility, using expected_generation CAS.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
import numpy as np
from .errors import KnowledgeError
from .chunking import split_pages, normalize_search_text
from .security import same_scope, require_id, validate_advisor_acl

SCHEMA = "knowledge-index-v1"
_FILES = {"manifest.json", "chunks.sqlite", "vectors.npy", "vectors.faiss"}
MAX_UNPACKED = 210 * 1024 * 1024
MAX_CHUNKS = 18000


@dataclass
class IndexPack:
    manifest: dict
    rows: list
    vectors: np.ndarray
    sqlite_blob: bytes
    faiss_blob: bytes | None = None


def _sqlite_blob(rows):
    fd, tmp = tempfile.mkstemp(prefix="athar_kb_", suffix=".sqlite")
    os.close(fd)
    try:
        conn = sqlite3.connect(tmp)
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, resource_id TEXT, resource_revision INTEGER, scope TEXT, organization_id TEXT, project_id TEXT, title TEXT, page_start INTEGER, page_end INTEGER, section TEXT, text TEXT, search_text TEXT)")
            conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(search_text, tokenize='unicode61 remove_diacritics 2')")
            for i, row in enumerate(rows, 1):
                conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (i, row["chunk_id"], row["resource_id"], row["resource_revision"], row["scope"], row.get("organization_id"), row.get("project_id"), row["title"], row.get("page_start"), row.get("page_end"), row.get("section", ""), row["text"], row["search_text"]))
                conn.execute("INSERT INTO chunks_fts (rowid,search_text) VALUES (?,?)", (i, row["search_text"]))
            conn.commit()
        finally:
            conn.close()
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


def _faiss_blob(vectors):
    try:
        import faiss
    except ImportError:
        return None  # NumPy exact search is supported in offline tests only.
    index = faiss.IndexFlatIP(vectors.shape[1])
    if len(vectors):
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return faiss.serialize_index(index).tobytes()


def serialize_pack(pack: IndexPack):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as out:
        out.writestr("manifest.json", json.dumps(pack.manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        out.writestr("chunks.sqlite", pack.sqlite_blob)
        np_buf = io.BytesIO()
        np.save(np_buf, pack.vectors, allow_pickle=False)
        out.writestr("vectors.npy", np_buf.getvalue())
        if pack.faiss_blob:
            out.writestr("vectors.faiss", pack.faiss_blob)
    return buf.getvalue()


def open_pack(raw: bytes, *, scope=None, organization_id=None, project_id=None, generation=None, model_id=None):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            if len(names) != len(set(names)) or not set(names).issubset(_FILES) or not {"manifest.json", "chunks.sqlite", "vectors.npy"}.issubset(names):
                raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Unexpected index archive files")
            if any(f.file_size > MAX_UNPACKED for f in z.infolist()) or sum(f.file_size for f in z.infolist()) > MAX_UNPACKED:
                raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Oversized index archive")
            manifest = json.loads(z.read("manifest.json"))
            sql_blob = z.read("chunks.sqlite")
            vectors = np.load(io.BytesIO(z.read("vectors.npy")), allow_pickle=False)
            faiss_blob = z.read("vectors.faiss") if "vectors.faiss" in names else None
    except KnowledgeError:
        raise
    except Exception:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Index archive unreadable") from None
    if not isinstance(manifest, dict) or manifest.get("index_schema_version") != SCHEMA:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Index schema mismatch")
    if scope is not None:
        same_scope(manifest, scope, organization_id, project_id)
    if generation is not None and manifest.get("generation") != generation:
        raise KnowledgeError("INDEX_STALE", "Index generation mismatch")
    if model_id is not None and manifest.get("embedding_model_id") != model_id:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Embedding model mismatch")
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.dtype != np.float32 or vectors.shape[0] > MAX_CHUNKS:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Invalid index vectors")
    if not np.isfinite(vectors).all() or vectors.shape[1] < 8 or vectors.shape[1] > 4096:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Invalid index vector values")
    if manifest.get("vector_dim") != vectors.shape[1] or manifest.get("chunk_count") != len(vectors):
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Vector count/dimension mismatch")
    # Only read a SQLite database after validating the signed artifact checksum at transfer.
    fd, tmp = tempfile.mkstemp(prefix="athar_kb_read_", suffix=".sqlite")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(sql_blob)
        # Immutable: do not permit any SQL writes while reading the imported database.
        conn = sqlite3.connect("file:" + tmp + "?immutable=1", uri=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            data = conn.execute("SELECT chunk_id, resource_id, resource_revision, scope, organization_id, project_id, title, page_start, page_end, section, text, search_text FROM chunks ORDER BY id").fetchall()
            rows = [dict(zip(("chunk_id","resource_id","resource_revision","scope","organization_id","project_id","title","page_start","page_end","section","text","search_text"), x)) for x in data]
            if len(rows) != len(vectors) or len(rows) > MAX_CHUNKS:
                raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Index chunk/vector mismatch")
        finally:
            conn.close()
    except KnowledgeError:
        raise
    except Exception:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Index SQL database invalid") from None
    finally:
        os.unlink(tmp)
    for row in rows:
        same_scope(row, manifest["scope"], manifest.get("organization_id"), manifest.get("project_id"))
        if row["resource_id"] not in manifest.get("resources", {}):
            raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Missing resource metadata")
    return IndexPack(manifest, rows, np.ascontiguousarray(vectors), sql_blob, faiss_blob)


def make_pack(*, previous: IndexPack | None, resource: dict, pages: list | None, embedding, generation: int, operation: str):
    scope = resource["scope"]
    org = resource.get("organization_id") if scope == "organization_library" else None
    project = resource.get("project_id") if scope == "organization_library" else None
    resource_id = require_id(resource.get("resource_id"), "resource_id")
    revision = resource.get("revision")
    if type(revision) is not int or revision < 1:
        raise KnowledgeError("INVALID_REQUEST", "Resource revision must be a positive integer")
    if type(generation) is not int or generation < 1:
        raise KnowledgeError("INVALID_REQUEST", "Invalid generation")
    if operation not in ("upsert", "delete"):
        raise KnowledgeError("INVALID_REQUEST", "Unsupported indexing operation")
    if previous:
        same_scope(previous.manifest, scope, org, project)
        if previous.manifest["embedding_model_id"] != embedding.id or previous.manifest["vector_dim"] != embedding.dim:
            raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Changed embedding configuration requires explicit migration")
        if previous.manifest["generation"] != generation-1:
            raise KnowledgeError("INDEX_STALE", "Expected generation mismatch")
        resources = dict(previous.manifest["resources"])
        existing = resources.get(resource_id)
        if existing and revision <= existing["revision"]:
            if revision == existing["revision"]:
                return None  # Idempotent repeated event; skip upload.
            raise KnowledgeError("INDEX_STALE", "Older resource revision rejected")
        keep = [i for i, row in enumerate(previous.rows) if row["resource_id"] != resource_id]
        rows = [previous.rows[i].copy() for i in keep]
        old_vectors = previous.vectors[keep]
    else:
        resources = {}
        existing = None
        rows = []
        old_vectors = np.empty((0, embedding.dim), dtype=np.float32)
        if generation != 1:
            raise KnowledgeError("INDEX_NOT_READY", "Previous index required for generation > 1")
    new_vectors = np.empty((0, embedding.dim), dtype=np.float32)
    if operation == "delete":
        if not previous or not existing:
            raise KnowledgeError("INDEX_NOT_READY", "Resource cannot be deleted from missing index")
        resources.pop(resource_id, None)
    else:
        title = resource.get("title", "")
        if not isinstance(title, str) or not title.strip() or len(title) > 300:
            raise KnowledgeError("INVALID_REQUEST", "Invalid resource title")
        meta = {"revision": revision, "title": title.strip(), "advisors": [], "include_meta_advisor": False, "chunks": 0}
        if scope == "advisor_library":
            meta["advisors"], meta["include_meta_advisor"] = validate_advisor_acl(resource)
            meta["layer"] = resource.get("layer", "central")
        if pages is None and existing:
            # ACL/title-only changes do not re-embed; retain the previous chunks.
            indices = [i for i, row in enumerate(previous.rows) if row["resource_id"] == resource_id]
            revived = [dict(previous.rows[i], resource_revision=revision, title=title.strip(), chunk_id=f"{resource_id}:rev-{revision}:chunk-{j:05d}") for j,i in enumerate(indices, 1)]
            rows.extend(revived)
            new_vectors = previous.vectors[indices]
        else:
            if pages is None:
                raise KnowledgeError("INVALID_REQUEST", "Missing document text/content")
            chunks = split_pages(pages)
            if not chunks:
                raise KnowledgeError("SOURCE_UNREADABLE", "No usable text chunks")
            new_vectors = embedding.embed([c["text"] for c in chunks], is_query=False)
            if new_vectors.shape != (len(chunks), embedding.dim):
                raise KnowledgeError("EMBEDDING_MODEL_UNAVAILABLE", "Embedding shape mismatch")
            for j, c in enumerate(chunks, 1):
                rows.append({
                    "chunk_id": f"{resource_id}:rev-{revision}:chunk-{j:05d}",
                    "resource_id": resource_id, "resource_revision": revision,
                    "scope": scope, "organization_id": org, "project_id": project,
                    "title": title.strip(), "page_start": c["page_start"], "page_end": c["page_end"],
                    "section": c["section"], "text": c["text"], "search_text": c["search_text"],
                })
        meta["chunks"] = len(rows) - len(old_vectors)
        resources[resource_id] = meta
    vectors = np.concatenate((old_vectors, new_vectors), axis=0).astype(np.float32, copy=False)
    if len(rows) > MAX_CHUNKS or len(rows) != len(vectors):
        raise KnowledgeError("DOCUMENT_TOO_LARGE", "Index has too many chunks")
    # Compaction: resources/chunks are immutable within this *new* generation.
    manifest = {
        "index_schema_version": SCHEMA, "scope": scope,
        "organization_id": org, "project_id": project,
        "generation": generation, "embedding_model_id": embedding.id,
        "vector_dim": embedding.dim, "chunk_count": len(rows),
        "resources": resources, "search_engine": "fts5+faiss-or-exact-dot",
    }
    return IndexPack(manifest, rows, np.ascontiguousarray(vectors), _sqlite_blob(rows), _faiss_blob(vectors))
