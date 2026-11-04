"""Deterministic page-aware chunking and Arabic-aware lexical normalisation."""
import re
import unicodedata
from .extraction import clean_text

ARABIC_MARKS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def normalize_search_text(text):
    text = ARABIC_MARKS.sub("", unicodedata.normalize("NFKC", str(text)).lower())
    return text.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ٱ":"ا", "ى":"ي", "ؤ":"و", "ئ":"ي"}))


def split_pages(pages, chunk_size=1000, overlap=140):
    if not (200 <= chunk_size <= 3000 and 0 <= overlap < chunk_size // 2):
        raise ValueError("Invalid chunk settings")
    result = []
    for page in pages:
        text = clean_text(page["text"])
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", pos + chunk_size//2, end),
                               text.rfind(". ", pos + chunk_size//2, end),
                               text.rfind("، ", pos + chunk_size//2, end))
                if boundary > pos + chunk_size//2:
                    end = boundary + 1
            segment = text[pos:end].strip()
            if segment:
                result.append({
                    "page_start": page.get("page_start"),
                    "page_end": page.get("page_end"),
                    "section": page.get("section") or "",
                    "text": segment,
                    "search_text": normalize_search_text(segment),
                })
            if end >= len(text):
                break
            pos = max(pos + 1, end - overlap)
    return result
