"""Local cross-encoder reranking (BAAI/bge-reranker-v2-m3)."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from backend.config import get_config
from backend.indexing.embedder import resolve_device

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    rcfg, icfg = get_config()["retrieval"], get_config()["indexing"]
    device = resolve_device(rcfg.get("reranker_device") or icfg.get("embedding_device"))
    model = CrossEncoder(rcfg["reranker_model"], max_length=rcfg.get("reranker_max_length", 512),
                         device=device)
    if device == "cuda" and icfg.get("embedding_fp16", True):
        model.model.half()
    # Decide ONCE whether predict() already returns probabilities. The old per-batch check
    # (`min < 0 or max > 1`) could leave one query's scores as logits and another's as
    # probabilities, which breaks the fixed-size timestamp boost.
    act = getattr(model, "activation_fn", None) or getattr(model, "activation_fct", None) \
        or getattr(model, "default_activation_function", None)
    name = type(act).__name__.lower() if act is not None else ""
    needs_sigmoid = "sigmoid" not in name
    log.info("Reranker %s on %s (activation=%s, sigmoid applied here=%s)",
             rcfg["reranker_model"], device, name or "unknown", needs_sigmoid)
    return model, needs_sigmoid


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance probabilities in 0..1, one per text."""
    if not texts:
        return []
    model, needs_sigmoid = _model()
    bs = get_config()["retrieval"].get("reranker_batch_size", 16)
    raw = np.asarray(model.predict([(query, t) for t in texts], batch_size=bs,
                                   show_progress_bar=False), dtype=float).reshape(-1)
    if needs_sigmoid:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.tolist()
