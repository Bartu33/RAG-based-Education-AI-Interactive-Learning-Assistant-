"""Retrieval tests that run without downloading BGE-m3 / the reranker: the embedder is replaced by a
deterministic bag-of-words hash embedding, the real ChromaDB store is used in a temp folder."""
import hashlib

import numpy as np
import pytest

from backend.retrieval import bm25, retriever
from backend.retrieval.retriever import rrf_fuse, timestamp_bonus
from backend.schemas import Segment

chromadb = pytest.importorskip("chromadb")
DIM = 64


def fake_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for tok in bm25.normalize(text, 5):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from backend.indexing.vectorstore import ChromaStore
    monkeypatch.setattr(retriever, "embed_query", lambda t: fake_embed(t).tolist())
    bm25.clear_cache()
    s = ChromaStore(collection="test_chunks", path=str(tmp_path), fresh=True)
    segs = [
        Segment("Nyquist örnekleme teoremi: örnekleme frekansı en yüksek frekansın iki katı olmalı.",
                "pdf", "kitap.pdf", page=10, course="CENG384", lecture=5),
        Segment("Aliasing, örnekleme frekansı yetersiz olduğunda spektrumun üst üste binmesidir.",
                "pptx", "slayt.pptx", page=4, course="CENG384", lecture=5),
        Segment("Fourier dönüşümü sinyali frekans bileşenlerine ayırır.",
                "pdf", "kitap.pdf", page=3, course="CENG384", lecture=2),
        Segment("Hoca burada örnekleme frekansını tahtada gösteriyor.", "video", "ders5.mp4",
                timestamp_start_ms=600_000, timestamp_end_ms=660_000, course="CENG384", lecture=5),
        Segment("Gereksinim analizi yazılım projesinin ilk aşamasıdır.",
                "docx", "notlar.docx", course="SENG449", lecture=1),
    ]
    s.add(segs, np.stack([fake_embed(x.text) for x in segs]))
    return s


def test_normalize_turkish():
    assert bm25.normalize("ÖRNEKLEME Işık İstanbul", 0) == ["ornekleme", "isik", "istanbul"]
    assert bm25.normalize("örneklemenin", 5) == bm25.normalize("ornekleme", 5)


def test_bm25_ranks_exact_term_first():
    docs = [{"id": str(i), "text": t, "metadata": {}} for i, t in
            enumerate(["Nyquist teoremi", "Fourier serisi", "Laplace dönüşümü"])]
    top = bm25.BM25Index(docs).top("nyquist nedir", 3)
    assert top[0]["id"] == "0" and len(top) == 1


def test_rrf_prefers_items_found_by_both():
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    b = [{"id": "z"}, {"id": "y"}]
    order = [h["id"] for h in rrf_fuse([a, b])]
    assert set(order) == {"x", "y", "z"}
    assert order[-1] == "x"                                    # found by only one list


def test_timestamp_bonus_decay():
    assert timestamp_bonus(0, 0.2, 60_000) == 0.2
    assert timestamp_bonus(30_000, 0.2, 60_000) == pytest.approx(0.1)
    assert timestamp_bonus(90_000, 0.2, 60_000) == 0.0
    assert timestamp_bonus(30_000, 0.2, 0) == 0.0
    assert timestamp_bonus(None, 0.2, 60_000) == 0.0


def test_course_and_lecture_filter(store):
    r = retriever.retrieve("örnekleme frekansı", "CENG384", 5, store=store, use_reranker=False)
    metas = [c["metadata"] for c in r["chunks"]]
    assert metas and all(m["course"] == "CENG384" and m["lecture"] == 5 for m in metas)
    assert r["filtered_from"] == 3


def test_hybrid_returns_lexical_match(store):
    r = retriever.retrieve("Nyquist", "CENG384", None, store=store, use_reranker=False,
                           hybrid=True, top_context=1)
    assert r["chunks"][0]["metadata"]["document"] == "kitap.pdf"
    assert r["mode"] == "hybrid" and "bm25_ms" in r["timings"]


def test_bm25_cache_invalidated_on_write(store):
    where = {"course": {"$eq": "SENG449"}}
    first = bm25.get_index(store, where)
    store.add([Segment("Test planı", "docx", "plan.docx", course="SENG449", lecture=1)],
              np.stack([fake_embed("Test planı")]))
    assert len(bm25.get_index(store, where)) == len(first) + 1


def test_timestamp_window_promotes_video(store):
    r = retriever.retrieve("örnekleme frekansı", "CENG384", 5, ts_range_ms=(590_000, 640_000),
                           store=store, use_reranker=False, timestamp_boost=1.0, top_context=1)
    top = r["chunks"][0]
    assert top["metadata"]["source_type"] == "video" and top["in_window"]


def test_consistency_table_detects_clusters():
    from scripts.retrieval_bench import consistency_table
    rng = np.random.default_rng(1)
    cent = rng.normal(0, 1, (3, 32))
    meta, emb = [], []
    for lec in range(3):
        for _ in range(10):
            meta.append({"course": "C", "lecture": lec + 1})
            emb.append(cent[lec] + rng.normal(0, 0.5, 32))
    df = consistency_table(meta, np.array(emb))
    assert len(df) == 3 and df["pass"].all() and (df["margin"] > 0).all()


def test_dedupe_keeps_best_and_lists_twins():
    from backend.retrieval.retriever import dedupe_hits
    meta = lambda p: {"source_type": "pdf", "document": "c1.pdf", "page": p}
    hits = [{"id": "a", "text": "Chapter 1: roadmap Introduction: 1-69 What is a protocol?", "metadata": meta(69)},
            {"id": "b", "text": "Packet switching versus circuit switching", "metadata": meta(39)},
            {"id": "c", "text": "Chapter 1: roadmap Introduction: 1-81 What is a protocol?", "metadata": meta(81)}]
    out = dedupe_hits(hits, 0.9)
    assert [h["id"] for h in out] == ["a", "b"]
    assert out[0]["duplicates"] == ["c1.pdf, p. 81"]


def test_reranker_abstains_when_all_scores_low(store, monkeypatch):
    from backend import config
    monkeypatch.setitem(config.get_config()["retrieval"], "rerank_min_score", 0.05)
    monkeypatch.setattr(retriever, "rerank_scores", lambda q, texts: [0.001] * len(texts))
    dense = retriever.retrieve("örnekleme frekansı", "CENG384", 5, store=store, use_reranker=False)
    r = retriever.retrieve("örnekleme frekansı", "CENG384", 5, store=store, use_reranker=True)
    assert r["mode"] == "dense+rerank(abstained)"
    assert [c["id"] for c in r["chunks"]] == [c["id"] for c in dense["chunks"]]   # dense order kept
    assert all(c["rerank"] == 0.001 for c in r["chunks"])


def test_reranker_used_when_confident(store, monkeypatch):
    monkeypatch.setattr(retriever, "rerank_scores",
                        lambda q, texts: [0.9 if "Aliasing" in t else 0.2 for t in texts])
    r = retriever.retrieve("örnekleme frekansı", "CENG384", 5, store=store, use_reranker=True)
    assert r["mode"] == "dense+rerank" and r["chunks"][0]["text"].startswith("Aliasing")
