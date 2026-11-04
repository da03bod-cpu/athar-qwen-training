"""Small multilingual E5 encoder on CPU; independent from the Qwen GPU.

Production image prefetches weights at Docker build time. No embedding inference
runs during normal analysis except a small *query* embedding.
"""
from __future__ import annotations
import os
import threading
import numpy as np
from .errors import KnowledgeError

EMBEDDING_MODEL = os.environ.get("ATHAR_KNOWLEDGE_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_EMBEDDER = None
_LOCK = threading.Lock()


class E5Embedder:
    def __init__(self, model_id=EMBEDDING_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModel
        offline = os.environ.get("ATHAR_KNOWLEDGE_OFFLINE", "1") == "1"
        self.id = model_id
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=offline).to("cpu").eval()
        self.dim = int(self.model.config.hidden_size)

    def embed(self, texts, is_query=False):
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = []
        prefix = "query: " if is_query else "passage: "
        with self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                part = [prefix + x for x in texts[start:start+16]]
                toks = self.tok(part, padding=True, truncation=True, max_length=512, return_tensors="pt")
                outputs = self.model(**toks).last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).to(outputs.dtype)
                pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normed = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normed.float().cpu().numpy())
        return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        with _LOCK:
            if _EMBEDDER is None:
                try:
                    _EMBEDDER = E5Embedder()
                except Exception:
                    # The real implementation must not silently fall back to bogus embeddings.
                    raise KnowledgeError("EMBEDDING_MODEL_UNAVAILABLE", "Local embedding model is unavailable") from None
    return _EMBEDDER
