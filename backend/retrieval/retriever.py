from time import perf_counter
from typing import Optional

from backend.config import get_config
from backend.indexing.embedder import embed_query
from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.retrieval.reranker import rerank_scores


def _in_window(meta: dict, window: Optional[tuple[int, int]]) -> bool:
    if not window or meta.get("source_type") != "video" or meta.get("timestamp_start_ms") is None:
        return False
    return meta["timestamp_start_ms"] <= window[1] and meta["timestamp_end_ms"] >= window[0]


def retrieve(question: str, course: str, lecture: Optional[int] = None,
             ts_range_ms: Optional[tuple[int, int]] = None,
             top_candidates: Optional[int] = None, top_context: Optional[int] = None,
             store: Optional[VectorStore] = None) -> dict:
    """Narrow -> search -> rerank -> (timestamp boost) -> top-k context."""
    cfg = get_config()["retrieval"]
    store = store or get_store()
    top_candidates = top_candidates or cfg["top_candidates"]
    top_context = top_context or cfg["top_context"]

    base = [{"course": {"$eq": course}}]
    if lecture:
        base.append({"lecture": {"$eq": int(lecture)}})
    where = and_filter(base)

    t0 = perf_counter()
    q = embed_query(question)
    hits = store.query(q, top_candidates, where)

    if ts_range_ms:  # also pull candidates from the video segment being watched
        win_where = and_filter(base + [
            {"source_type": {"$eq": "video"}},
            {"timestamp_end_ms": {"$gte": ts_range_ms[0]}},
            {"timestamp_start_ms": {"$lte": ts_range_ms[1]}},
        ])
        seen = {h["id"] for h in hits}
        hits += [h for h in store.query(q, cfg["window_extra_candidates"], win_where)
                 if h["id"] not in seen]

    if hits and cfg["use_reranker"]:
        for h, s in zip(hits, rerank_scores(question, [h["text"] for h in hits])):
            h["similarity"], h["score"] = h["score"], s

    for h in hits:
        h["in_window"] = _in_window(h["metadata"], ts_range_ms)
        if h["in_window"]:
            h["score"] += cfg["timestamp_boost"]

    hits.sort(key=lambda h: h["score"], reverse=True)
    retrieval_ms = int((perf_counter() - t0) * 1000)

    return {"chunks": hits[:top_context], "candidates": len(hits),
            "filtered_from": store.count(where), "retrieval_ms": retrieval_ms}