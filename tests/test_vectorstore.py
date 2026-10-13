"""ChromaDB store + indexer tests (real ChromaDB in a temp folder, no model download)."""
import numpy as np
import pytest

from backend.schemas import Segment

pytest.importorskip("chromadb")
from backend.indexing import indexer  # noqa: E402
from backend.indexing.vectorstore import (ChromaStore, EmbeddingModelMismatch,  # noqa: E402
                                          cleanup_collections)

DIM = 8


def vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def segs():
    return [
        Segment("Nyquist", "pdf", "kitap.pdf", page=1, course="CENG384", lecture=5),
        Segment("Aliasing", "pptx", "slayt.pptx", page=2, course="CENG384", lecture=5),
        Segment("Fourier", "pdf", "kitap.pdf", page=9, course="CENG384", lecture=2),
        Segment("hoca anlatıyor", "video", "d5.mp4", timestamp_start_ms=0, timestamp_end_ms=9000,
                confidence=0.42, course="CENG384", lecture=5),
        Segment("net ses", "video", "d5.mp4", timestamp_start_ms=9000, timestamp_end_ms=18000,
                confidence=0.91, course="CENG384", lecture=5),
        Segment("Gereksinim", "docx", "not.docx", course="SENG449", lecture=1),
    ]


@pytest.fixture()
def store(tmp_path):
    s = ChromaStore(collection="test_store", path=str(tmp_path), fresh=True)
    items = segs()
    s.add(items, np.stack([vec(i) for i in range(len(items))]))
    return s


def test_list_courses_and_stats(store):
    courses = {c["course"]: c for c in store.list_courses()}
    assert courses["CENG384"]["lectures"] == [2, 5] and courses["CENG384"]["chunks"] == 5
    assert courses["SENG449"]["documents"] == 1
    st = store.stats()
    assert st["chunks"] == 6 and st["courses"] == 2 and st["by_type"]["video"] == 2
    assert st["embedding_model"] == "BAAI/bge-m3"


def test_delete_course_and_lecture(store):
    v0 = store.version
    assert store.delete_course("CENG384", 2) == 1
    assert store.count({"course": {"$eq": "CENG384"}}) == 4
    assert store.delete_course("SENG449") == 1
    assert store.delete_course("NOPE") == 0
    assert store.version > v0


def test_delete_where_requires_filter(store):
    with pytest.raises(ValueError):
        store.delete_where(None)


def test_clear_keeps_collection_usable(store):
    store.clear()
    assert store.count() == 0
    store.add(segs()[:1], np.stack([vec(0)]))
    assert store.count() == 1


def test_low_confidence_only_video_below_threshold(store):
    low = store.low_confidence(0.6)
    assert [h["metadata"]["confidence"] for h in low] == [0.42]
    assert store.low_confidence(0.6, course="SENG449") == []


def test_update_text_marks_corrected(store):
    cid = store.low_confidence(0.6)[0]["id"]
    store.update_text(cid, "hoca Nyquist oranını anlatıyor", vec(99))
    hit = store.get_by_ids([cid])[0]
    assert hit["text"].startswith("hoca Nyquist")
    assert hit["metadata"]["corrected"] is True and hit["metadata"]["confidence"] == 1.0
    assert hit["metadata"]["timestamp_start_ms"] == 0          # metadata kept
    assert store.low_confidence(0.6) == []


def test_model_mismatch_is_refused(tmp_path, monkeypatch):
    ChromaStore(collection="guarded", path=str(tmp_path), fresh=True)
    from backend import config
    cfg = config.get_config()
    monkeypatch.setitem(cfg["indexing"], "embedding_model", "intfloat/multilingual-e5-large")
    with pytest.raises(EmbeddingModelMismatch):
        ChromaStore(collection="guarded", path=str(tmp_path))
    ChromaStore(collection="guarded", path=str(tmp_path), allow_model_change=True)  # inspect only


def test_cleanup_collections(tmp_path):
    ChromaStore(collection="bench_1000", path=str(tmp_path), fresh=True)
    ChromaStore(collection="keep_me", path=str(tmp_path), fresh=True)
    assert cleanup_collections("bench_", path=str(tmp_path)) == ["bench_1000"]


def test_index_file_progress_and_correction(tmp_path, monkeypatch):
    store = ChromaStore(collection="test_index", path=str(tmp_path), fresh=True)
    monkeypatch.setattr(indexer, "load_file", lambda p: [
        Segment("bir iki üç dört beş", "video", "ders.mp4", timestamp_start_ms=0,
                timestamp_end_ms=5000, confidence=0.3)])
    monkeypatch.setattr(indexer, "source_type_of", lambda p: "video")
    monkeypatch.setattr(indexer, "embed_texts", lambda texts: np.stack([vec(len(t)) for t in texts]))
    events = []
    rec = indexer.index_file(tmp_path / "ders.mp4", "CENG384", 3, store=store,
                             on_progress=lambda s, f: events.append((s, f)))
    assert rec["chunks"] == 1 and rec["low_confidence_segments"] == 1
    assert set(rec["timings_ms"]) == {"extract", "chunk", "embed", "store"}
    fractions = [f for _, f in events]
    assert fractions == sorted(fractions) and events[-1] == ("done", 1.0)

    rec2 = indexer.index_file(tmp_path / "ders.mp4", "CENG384", 3, store=store)  # re-index
    assert rec2["chunks"] == 1 and store.count() == 1                         # no duplicates

    cid = store.get()[0]["id"]
    fixed = indexer.correct_chunk(cid, "  düzeltilmiş metin ", store=store)
    assert fixed["text"] == "düzeltilmiş metin" and fixed["metadata"]["corrected"]
    with pytest.raises(ValueError):
        indexer.correct_chunk(cid, "   ", store=store)
