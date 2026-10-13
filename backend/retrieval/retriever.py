"""Narrow -> search (dense, optionally + BM25) -> rerank -> timestamp boost -> top-k context."""
from __future__ import annotations

import re
from time import perf_counter
from typing import Optional

from backend.config import get_config
from backend.indexing.embedder import embed_query
from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.retrieval import bm25
from backend.retrieval.reranker import rerank_scores
from backend.schemas import format_location


def _gap_ms(meta: dict, window: Optional[tuple[int, int]]) -> Optional[int]:
    """Distance in ms between a video chunk and the watched window (0 = overlap).
    None for non-video chunks or when no window is given."""
    if not window or meta.get("source_type") != "video" or meta.get("timestamp_start_ms") is None:
        return None
    start, end = meta["timestamp_start_ms"], meta["timestamp_end_ms"]
    if start <= window[1] and end >= window[0]:
        return 0
    return window[0] - end if end < window[0] else start - window[1]


def timestamp_bonus(gap: Optional[int], boost: float, decay_ms: int) -> float:
    """Full boost inside the window, linearly decaying to 0 at `decay_ms` outside it.
    decay_ms = 0 reproduces the original hard in/out behaviour."""
    if gap is None or boost <= 0:
        return 0.0
    if gap == 0:
        return boost
    if decay_ms <= 0:
        return 0.0
    return boost * max(0.0, 1.0 - gap / decay_ms)


_DIGITS = re.compile(r"\d+")
_WORDS = re.compile(r"\w+", re.UNICODE)


def _shape(text: str) -> frozenset:
    """Word set with numbers removed: slides repeated with only a page number changed match."""
    return frozenset(_WORDS.findall(_DIGITS.sub(" ", text.lower())))


def dedupe_hits(hits: list[dict], threshold: float = 0.9) -> list[dict]:
    """Drop near-duplicate chunks (Jaccard word overlap >= threshold), keeping the best-scored one.
    The kept hit lists the dropped locations in `duplicates` (e.g. the same roadmap slide on
    pages 28, 49, 69). Input must already be sorted best first."""
    kept: list[tuple[frozenset, dict]] = []
    for h in hits:
        shape = _shape(h["text"])
        twin = None
        for ks, kh in kept:
            union = len(shape | ks)
            if union and len(shape & ks) / union >= threshold:
                twin = kh
                break
        if twin is None:
            kept.append((shape, h))
        else:
            twin.setdefault("duplicates", []).append(format_location(h["metadata"]))
    return [h for _, h in kept]


def rrf_fuse(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score = sum 1 / (k + rank)."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, 1):
            entry = fused.setdefault(h["id"], {**h, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
            if "similarity" in h and "similarity" not in entry:
                entry["similarity"] = h["similarity"]
    return sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)


def retrieve(question: str, course: str, lecture: Optional[int] = None,
             ts_range_ms: Optional[tuple[int, int]] = None,
             top_candidates: Optional[int] = None, top_context: Optional[int] = None,
             store: Optional[VectorStore] = None, hybrid: Optional[bool] = None,
             use_reranker: Optional[bool] = None,
             timestamp_boost: Optional[float] = None) -> dict:
    """The optional keyword arguments override config.yaml (used by experiments and the UI)."""
    cfg = get_config()["retrieval"]
    store = store or get_store()
    top_candidates = top_candidates or cfg["top_candidates"]
    top_context = top_context or cfg["top_context"]
    hybrid = cfg.get("hybrid", False) if hybrid is None else hybrid
    use_reranker = cfg["use_reranker"] if use_reranker is None else use_reranker
    boost = cfg["timestamp_boost"] if timestamp_boost is None else timestamp_boost
    decay = int(cfg.get("timestamp_decay_ms", 0))
    timings: dict[str, int] = {}

    base = [{"course": {"$eq": course}}]
    if lecture:
        base.append({"lecture": {"$eq": int(lecture)}})
    where = and_filter(base)

    t0 = perf_counter()
    q = embed_query(question)
    t1 = perf_counter()
    timings["embed_ms"] = int((t1 - t0) * 1000)

    dense = store.query(q, top_candidates, where)
    for h in dense:
        h["similarity"] = h["score"]

    window_hits: list[dict] = []
    if ts_range_ms:  # also pull candidates from the video segment being watched
        win_where = and_filter(base + [
            {"source_type": {"$eq": "video"}},
            {"timestamp_end_ms": {"$gte": ts_range_ms[0]}},
            {"timestamp_start_ms": {"$lte": ts_range_ms[1]}},
        ])
        for h in store.query(q, cfg["window_extra_candidates"], win_where):
            h["similarity"] = h["score"]
            window_hits.append(h)
    t2 = perf_counter()
    timings["search_ms"] = int((t2 - t1) * 1000)

    if hybrid:
        index = bm25.get_index(store, where, cfg.get("bm25_prefix_len", 5))
        lexical = index.top(question, cfg.get("bm25_candidates", top_candidates))
        fused = rrf_fuse([dense, lexical], k=cfg.get("rrf_k", 60))
        hits = fused[:top_candidates]
        max_rrf = hits[0]["rrf"] if hits else 1.0
        for h in hits:
            h["score"] = h["rrf"] / max_rrf
        timings["bm25_ms"] = int((perf_counter() - t2) * 1000)
    else:
        hits = dense

    seen = {h["id"] for h in hits}
    hits += [h for h in window_hits if h["id"] not in seen]

    t3 = perf_counter()
    rerank_note = ""
    if hits and use_reranker:
        first_stage = [h["score"] for h in hits]
        scores = rerank_scores(question, [h["text"] for h in hits])
        if max(scores) < cfg.get("rerank_min_score", 0.0):
            # The cross-encoder rejects every candidate. Seen with Turkish questions over English
            # slides: all scores ~0.00x, so their order is noise while the dense order is right.
            # Keep the first-stage order and say so in `mode`.
            for h, s in zip(hits, scores):
                h["rerank"] = s
            rerank_note = "(abstained)"
        else:
            for h, s, f in zip(hits, scores, first_stage):
                h["rerank"], h["score"] = s, s
        timings["rerank_ms"] = int((perf_counter() - t3) * 1000)

    for h in hits:
        gap = _gap_ms(h["metadata"], ts_range_ms)
        h["in_window"] = gap == 0
        h["score"] += timestamp_bonus(gap, boost, decay)

    hits.sort(key=lambda h: h["score"], reverse=True)
    dedupe = cfg.get("dedupe_threshold", 0.0)
    if dedupe and dedupe > 0:
        hits = dedupe_hits(hits, dedupe)
    retrieval_ms = int((perf_counter() - t0) * 1000)

    return {"chunks": hits[:top_context], "candidates": len(hits),
            "filtered_from": store.count(where), "retrieval_ms": retrieval_ms,
            "timings": timings, "mode": ("hybrid" if hybrid else "dense")
            + ("+rerank" + rerank_note if use_reranker else "")}
