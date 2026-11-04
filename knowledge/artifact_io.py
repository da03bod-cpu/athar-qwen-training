"""Backend-issued HTTPS blob transfer, with strict integrity and no redirects."""
from __future__ import annotations
import urllib.request
import urllib.error
from .errors import KnowledgeError
from .security import validate_backend_url, verify_bytes

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_INDEX_BYTES = 160 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise KnowledgeError("UNTRUSTED_REFERENCE", "Redirect forbidden")


_OPENER = urllib.request.build_opener(NoRedirect())


def download(ref: dict, *, max_bytes=MAX_INDEX_BYTES) -> bytes:
    if not isinstance(ref, dict):
        raise KnowledgeError("INVALID_REQUEST", "Missing authorized read reference")
    url = validate_backend_url(ref.get("download_url"))
    expected = ref.get("sha256")
    if not expected:
        raise KnowledgeError("INVALID_REQUEST", "Reference missing SHA-256")
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    try:
        with _OPENER.open(req, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise KnowledgeError("DOCUMENT_TOO_LARGE")
            chunks = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise KnowledgeError("DOCUMENT_TOO_LARGE")
                chunks.append(chunk)
        payload = b"".join(chunks)
        verify_bytes(payload, expected, max_bytes, ref.get("size_bytes"))
        return payload
    except KnowledgeError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise KnowledgeError("SIGNED_REF_EXPIRED", "Authorization failed or reference expired") from None
        raise KnowledgeError("ARTIFACT_TRANSFER_FAILED", "Backend file download failed") from None
    except (ValueError, OSError, urllib.error.URLError):
        raise KnowledgeError("ARTIFACT_TRANSFER_FAILED", "Backend file download unavailable") from None


def upload(ref: dict, payload: bytes) -> None:
    if not isinstance(ref, dict) or not ref.get("upload_url"):
        raise KnowledgeError("INVALID_REQUEST", "Missing authorized write reference")
    if len(payload) > MAX_INDEX_BYTES:
        raise KnowledgeError("DOCUMENT_TOO_LARGE", "Index too large")
    url = validate_backend_url(ref["upload_url"])
    method = ref.get("method", "PUT")
    if method != "PUT":
        raise KnowledgeError("INVALID_REQUEST", "Artifact upload must use PUT")
    req = urllib.request.Request(url, data=payload, method="PUT", headers={"Content-Type": "application/octet-stream"})
    try:
        with _OPENER.open(req, timeout=60) as response:
            if response.status not in (200, 201, 202, 204):
                raise KnowledgeError("ARTIFACT_TRANSFER_FAILED")
    except KnowledgeError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise KnowledgeError("SIGNED_REF_EXPIRED") from None
        if exc.code in (409, 412):
            raise KnowledgeError("INDEX_CONFLICT") from None
        raise KnowledgeError("ARTIFACT_TRANSFER_FAILED", "Backend upload failed") from None
    except (OSError, urllib.error.URLError):
        raise KnowledgeError("ARTIFACT_TRANSFER_FAILED", "Backend upload unavailable") from None
