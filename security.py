"""Fail-closed request, tenant, and authorized-reference checks."""
from __future__ import annotations
import hashlib
import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit
from .errors import KnowledgeError

HEX64 = re.compile(r"^[a-fA-F0-9]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SCOPES = {"advisor_library", "organization_library"}


def require_id(value, name):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise KnowledgeError("INVALID_REQUEST", f"Invalid {name}")
    return value


def scope_from_resource(resource):
    if not isinstance(resource, dict):
        raise KnowledgeError("INVALID_REQUEST", "resource must be an object")
    scope = resource.get("scope")
    if scope not in SCOPES:
        raise KnowledgeError("INVALID_REQUEST", "Unknown knowledge scope")
    organization_id = project_id = None
    if scope == "organization_library":
        organization_id = require_id(resource.get("organization_id"), "organization_id")
        project_id = require_id(resource.get("project_id"), "project_id")
    elif resource.get("organization_id") or resource.get("project_id"):
        raise KnowledgeError("TENANT_SCOPE_MISMATCH", "Advisor scope cannot contain organization/project")
    return scope, organization_id, project_id


def same_scope(manifest, scope, organization_id, project_id):
    if (manifest.get("scope") != scope
        or manifest.get("organization_id") != organization_id
        or manifest.get("project_id") != project_id):
        raise KnowledgeError("TENANT_SCOPE_MISMATCH", "Index scope does not match requested scope")


def valid_sha256(value):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise KnowledgeError("INVALID_REQUEST", "A valid sha256 is required")
    return value.lower()


def _allowed_hosts():
    hosts = os.environ.get("KNOWLEDGE_ALLOWED_HOSTS", "")
    return {h.strip().lower() for h in hosts.split(",") if h.strip()}


def validate_backend_url(url: str):
    # This URL is passed only by a trusted Backend integration; not by end users.
    # Hostname allowlist and DNS checks provide a second layer of SSRF defense.
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.fragment:
            raise ValueError()
        if parsed.port not in (None, 443):
            raise ValueError()
        hosts = _allowed_hosts()
        if not hosts or host not in hosts:
            raise KnowledgeError("UNTRUSTED_REFERENCE", "Reference host not allowlisted")
        # Reject IP literals and hostnames resolving to loopback/private/link-local.
        try:
            ipaddress.ip_address(host)
            raise KnowledgeError("UNTRUSTED_REFERENCE", "IP literal not allowed")
        except ValueError:
            pass
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not infos:
            raise ValueError()
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise KnowledgeError("UNTRUSTED_REFERENCE", "Reference address not public")
    except KnowledgeError:
        raise
    except (TypeError, ValueError, OSError):
        raise KnowledgeError("UNTRUSTED_REFERENCE", "Invalid or unresolved backend reference") from None
    return url


def verify_bytes(payload: bytes, expected_digest: str, max_bytes: int, expected_size=None):
    if len(payload) > max_bytes:
        raise KnowledgeError("DOCUMENT_TOO_LARGE", "Artifact exceeds maximum size")
    if expected_size is not None and len(payload) != expected_size:
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Artifact size mismatch")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != valid_sha256(expected_digest):
        raise KnowledgeError("INDEX_INTEGRITY_FAILED", "Artifact checksum mismatch")
    return actual


def advisor_permitted(row, advisor_id=None, is_meta=False):
    if is_meta:
        return row.get("include_meta_advisor") is True
    if not advisor_id or not isinstance(advisor_id, str):
        return False
    allowed = row.get("advisors", [])
    return allowed == "all" or (isinstance(allowed, list) and advisor_id in allowed)


def validate_advisor_acl(resource):
    advisors = resource.get("advisors", [])
    meta = resource.get("include_meta_advisor", False)
    if not isinstance(meta, bool):
        raise KnowledgeError("INVALID_REQUEST", "include_meta_advisor must be boolean")
    if advisors != "all":
        if not isinstance(advisors, list) or not all(isinstance(x, str) and ID_RE.fullmatch(x) for x in advisors):
            raise KnowledgeError("INVALID_REQUEST", "Invalid advisor allowlist")
        if len(advisors) != len(set(advisors)):
            raise KnowledgeError("INVALID_REQUEST", "Duplicate advisor allowlist entries")
        if not advisors and not meta:
            raise KnowledgeError("INVALID_REQUEST", "At least one advisor or Meta permission required")
    return advisors, meta
