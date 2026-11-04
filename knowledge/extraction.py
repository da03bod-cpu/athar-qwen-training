"""Text extraction (no OCR); page/source metadata never leaves the scoped pack."""
from __future__ import annotations
import io
import re
import unicodedata
from .errors import KnowledgeError
from .artifact_io import MAX_SOURCE_BYTES

ALLOWED_MIMES = {
    "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain", "text/markdown", "text/csv",
}


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_pages(raw: bytes, mime_type: str):
    if mime_type not in ALLOWED_MIMES or not isinstance(raw, bytes):
        raise KnowledgeError("SOURCE_UNSUPPORTED", "Unsupported file format")
    if len(raw) > MAX_SOURCE_BYTES:
        raise KnowledgeError("DOCUMENT_TOO_LARGE")
    try:
        if mime_type == "application/pdf":
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            try:
                if doc.page_count > 600:
                    raise KnowledgeError("DOCUMENT_TOO_LARGE", "PDF page count exceeds limit")
                result = []
                for n, page in enumerate(doc, 1):
                    txt = clean_text(page.get_text("text"))
                    if txt:
                        result.append({"page_start": n, "page_end": n, "section": "", "text": txt})
            finally:
                doc.close()
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            d = Document(io.BytesIO(raw))
            blocks = [p.text for p in d.paragraphs if p.text.strip()]
            for table in d.tables:
                for row in table.rows:
                    blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
            result = [{"page_start": None, "page_end": None, "section": "", "text": clean_text("\n".join(blocks))}]
        else:
            txt = raw.decode("utf-8-sig", errors="strict")
            result = [{"page_start": None, "page_end": None, "section": "", "text": clean_text(txt)}]
    except KnowledgeError:
        raise
    except Exception:
        raise KnowledgeError("SOURCE_UNREADABLE", "Could not extract readable text") from None
    if not result or sum(len(p["text"]) for p in result) < 20:
        raise KnowledgeError("SOURCE_UNREADABLE", "No meaningful text layer; OCR is not enabled")
    if sum(len(p["text"]) for p in result) > 3_000_000:
        raise KnowledgeError("DOCUMENT_TOO_LARGE", "Extracted document text exceeds limit")
    return result
