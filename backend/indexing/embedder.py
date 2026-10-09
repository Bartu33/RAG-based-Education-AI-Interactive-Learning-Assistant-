from functools import lru_cache

import numpy as np

from backend.config import get_config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    cfg = get_config()["indexing"]
    model = SentenceTransformer(cfg["embedding_model"])  # picks GPU automatically if available
    model.max_seq_length = cfg["max_seq_length"]         # bge-m3 supports 8192; keep lower for speed
    return model


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    cfg = get_config()["indexing"]
    return _model().encode(texts, batch_size=batch_size or cfg["batch_size"],
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()