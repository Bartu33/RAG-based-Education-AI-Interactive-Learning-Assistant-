from functools import lru_cache

import numpy as np

from backend.config import get_config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(get_config()["retrieval"]["reranker_model"], max_length=512)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance scores in 0..1 (sigmoid applied if the model returns raw logits)."""
    if not texts:
        return []
    raw = np.asarray(_model().predict([(query, t) for t in texts], batch_size=8),
                     dtype=float).reshape(-1)
    if raw.min() < 0 or raw.max() > 1:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.tolist()