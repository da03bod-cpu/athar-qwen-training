"""Text extraction (no OCR); page/source metadata never leaves the scoped pack."""
from __future__ import annotations
import io
import math
import os
import re
import shutil
import subprocess
import tempfile
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



def _pdf_poppler_spread_text(pdf_path: str, page_number: int, rect):
    """Read a 2-up, wide PDF as separate right/left panels.

    Many RTL designed annual reports have a usable but poorly ordered PyMuPDF
    text layer. Poppler layout mode respects LTR digit runs (e.g. 200,000,000).
    This is text extraction, NOT OCR; unreadable scans still fail closed.
    """
    # At Poppler's default 72 DPI, x/y/W/H correspond to PDF point units.
    width, height = int(math.ceil(rect.width)), int(math.ceil(rect.height))
    midpoint = int(round(rect.width / 2.0))
    result = []
    for side, x, panel_width in (
        ("right", midpoint, max(1, width - midpoint)),
        ("left", 0, midpoint),
    ):
        cmd = ["pdftotext", "-layout", "-enc", "UTF-8", "-nopgbrk",
               "-f", str(page_number), "-l", str(page_number),
               "-x", str(x), "-y", "0", "-W", str(panel_width),
               "-H", str(height), pdf_path, "-"]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        text = clean_text(proc.stdout.decode("utf-8", errors="replace"))
        if text:
            result.append({"page_start": page_number, "page_end": page_number,
                           "section": "right panel" if side == "right" else "left panel",
                           "text": text})
    return result


def _pdf_extract_pages(raw: bytes):
    import fitz
    doc = fitz.open(stream=raw, filetype="pdf")
    if doc.page_count > 600:
        doc.close()
        raise KnowledgeError("DOCUMENT_TOO_LARGE", "PDF page count exceeds limit")

    # Poppler is only invoked for very wide Arabic spreads. Normal PDFs retain
    # the existing PyMuPDF behavior. The temporary file is private and deleted.
    tmp_path = None
    result = []
    poppler = shutil.which("pdftotext")
    try:
        for n, page in enumerate(doc, 1):
            raw_text = page.get_text("text")
            is_spread = (
                poppler is not None and page.rect.height > 0
                and page.rect.width / page.rect.height >= 2.1
                and re.search(r"[\u0600-\u06ff]", raw_text) is not None
            )
            if is_spread:
                if tmp_path is None:
                    fd, tmp_path = tempfile.mkstemp(prefix="athar_pdf_", suffix=".pdf")
                    with os.fdopen(fd, "wb") as f:
                        f.write(raw)
                panels = _pdf_poppler_spread_text(tmp_path, n, page.rect)
                if panels is not None and sum(len(p["text"]) for p in panels) >= 20:
                    result.extend(panels)
                    continue
            txt = clean_text(raw_text)
            if txt:
                result.append({"page_start": n, "page_end": n,
                               "section": "", "text": txt})
    finally:
        doc.close()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return result

def extract_pages(raw: bytes, mime_type: str):
    if not isinstance(raw, bytes):
        raise KnowledgeError("SOURCE_UNREADABLE", "Invalid source bytes")
    if mime_type not in ALLOWED_MIMES:
        # The Backend accepts uploads and marks unsupported formats as failed
        # asynchronously. A spreadsheet is not silently treated as CSV.
        raise KnowledgeError("SOURCE_UNREADABLE", "No supported text extraction for this file format")
    if len(raw) > MAX_SOURCE_BYTES:
        raise KnowledgeError("DOCUMENT_TOO_LARGE")
    try:
        if mime_type == "application/pdf":
            result = _pdf_extract_pages(raw)
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
