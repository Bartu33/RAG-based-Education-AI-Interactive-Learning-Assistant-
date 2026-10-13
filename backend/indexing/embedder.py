"""Local embeddings with sentence-transformers (BGE-m3 by default). Raw text never leaves the machine."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from backend.config import get_config

log = logging.getLogger(__name__)


def resolve_device(setting: Optional[str] = None) -> str:
    """'auto' -> cuda if available, else mps (Apple), else cpu. Explicit values pass through."""
    setting = (setting or "auto").lower()
    if setting != "auto":
        return setting
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    cfg = get_config()["indexing"]
    device = resolve_device(cfg.get("embedding_device"))
    model = SentenceTransformer(cfg["embedding_model"], device=device)
    model.max_seq_length = cfg["max_seq_length"]         # bge-m3 supports 8192; keep lower for speed
    if device == "cuda" and cfg.get("embedding_fp16", True):
        model.half()                                     # ~2x faster, half the VRAM, same ranking
    log.info("Embedding model %s on %s", cfg["embedding_model"], device)
    return model


def device_info() -> dict:
    """Used by the retrieval report so measurements state the hardware they ran on."""
    cfg = get_config()["indexing"]
    info = {"model": cfg["embedding_model"], "device": resolve_device(cfg.get("embedding_device")),
            "fp16": False, "gpu": None}
    try:
        import torch
        if info["device"] == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
            info["fp16"] = bool(cfg.get("embedding_fp16", True))
        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def _batch_size(batch_size: Optional[int]) -> int:
    cfg = get_config()["indexing"]
    if batch_size:
        return batch_size
    on_gpu = resolve_device(cfg.get("embedding_device")) == "cuda"
    return cfg.get("batch_size_gpu", 64) if on_gpu else cfg["batch_size"]


def embed_texts(texts: list[str], batch_size: int | None = None,
                show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 vectors, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    emb = _model().encode(texts, batch_size=_batch_size(batch_size),
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=show_progress)
    return np.asarray(emb, dtype=np.float32)               # fp16 model -> back to float32


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()
