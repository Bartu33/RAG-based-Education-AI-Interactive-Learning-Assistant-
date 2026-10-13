"""The UI's HTTP client (rag_ui/core/api_client.py) against the real FastAPI app with a fake store.
Checks that every UI call reaches an endpoint and gets the keys the views read."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402

from rag_ui.core import api_client as api  # noqa: E402

DOCS = [{"document": "Chapter_1.pdf", "course": "CENG336", "lecture": 1, "type": "pdf", "chunks": 93},
        {"document": "L1.mp4", "course": "CENG336", "lecture": 1, "type": "video", "chunks": 4}]


class FakeStore:
    def __init__(self):
        self.deleted, self.cleared = None, False

    def list_documents(self):
        return DOCS

    def list_courses(self):
        return [{"course": "CENG336", "lectures": [1], "documents": 2, "chunks": 97}]

    def stats(self):
        return {"collection": "c", "chunks": 97, "by_type": {"pdf": 93, "video": 4}, "courses": 1,
                "embedding_model": "BAAI/bge-m3", "hnsw": {}}

    def delete_course(self, course, lecture=None):
        self.deleted = (course, lecture)
        return 4

    def clear(self):
        self.cleared = True

    def low_confidence(self, threshold, course=None, limit=None):
        meta = {"document": "L1.mp4", "course": "CENG336", "lecture": 1, "timestamp_start_ms": 60_000,
                "timestamp_end_ms": 90_000, "confidence": 0.41}
        return [{"id": "abc", "text": "the nyquist rate", "metadata": meta, "score": 0.0}]


@pytest.fixture()
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(main, "get_store", lambda: s)
    api.use_client(TestClient(main.app))
    yield s
    api.use_client(None)


def test_status_and_config(store):
    assert api.health()["status"] == "ok"
    cfg = api.public_config()
    assert {"embedding_model", "chunk_size", "grounded_threshold", "benchmark_sizes"} <= cfg.keys()
    assert api.stats()["chunks"] == 97


def test_courses_and_documents(store):
    assert api.courses()[0]["lectures"] == [1]
    assert api.documents()[0]["chunks"] == 93


def test_delete_and_clear(store):
    assert api.delete_course("CENG336", 1) == 4 and store.deleted == ("CENG336", 1)
    assert api.delete_course("CENG336") == 4 and store.deleted == ("CENG336", None)
    api.clear_index()
    assert store.cleared


def test_low_confidence_and_correction(store, monkeypatch):
    seg = api.low_confidence(0.6, "CENG336")[0]
    assert seg["id"] == "abc" and seg["start_ms"] == 60_000 and seg["confidence"] == 0.41

    def fake_correct(chunk_id, text):
        if chunk_id != "abc":
            raise KeyError(chunk_id)
        return {"id": chunk_id, "text": text, "metadata": {"confidence": 1.0}, "score": 0.0}
    monkeypatch.setattr(main, "correct_chunk", fake_correct)
    assert api.correct_chunk("abc", "the Nyquist rate")["confidence"] == 1.0
    with pytest.raises(api.ApiError) as err:
        api.correct_chunk("nope", "x")
    assert err.value.status == 404


def test_ingest_upload(store, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "resolve_path", lambda p: tmp_path)
    monkeypatch.setattr(main, "index_file", lambda path, course, lecture: {
        "document": path.name, "course": course, "lecture": lecture, "type": "pdf", "chunks": 7,
        "low_confidence_segments": 0})
    rec = api.ingest_file("Chapter_1.pdf", b"%PDF-1.4", "CENG336", 2, "application/pdf")
    assert rec["chunks"] == 7 and rec["lecture"] == 2


def test_ask_sends_only_backend_settings(store, monkeypatch):
    import backend.tasks.qa as qa
    seen = {}

    def fake_answer(question, course, lecture, window, cfg):
        seen.update(lecture=lecture, window=window, cfg=cfg)
        return {"answer": "a [1]", "faithfulness": None, "sources": [], "contexts": [],
                "retrieval_ms": 1, "total_s": 0.1, "tokens_sent": 0, "candidates": 0, "filtered_from": 0}
    monkeypatch.setattr(qa, "answer_question", fake_answer)
    ui_cfg = {"provider": "Groq (Llama 3)", "top_k_candidates": 10, "top_k_context": 3, "th_grounded": 0.85}
    r = api.ask("q", "CENG336", None, (12, 14), ui_cfg)
    assert r["faithfulness"] is None and "contexts" not in r
    assert seen["lecture"] is None and seen["window"] == (720_000, 840_000)
    assert seen["cfg"] == {"provider": "Groq (Llama 3)", "top_k_candidates": 10, "top_k_context": 3}


def test_backend_errors_become_api_errors(store, monkeypatch):
    import backend.tasks.evaluator as ev

    def boom(*a, **k):
        raise ValueError("No indexed material found for this question.")
    monkeypatch.setattr(ev, "evaluate_answer", boom)
    with pytest.raises(api.ApiError) as err:
        api.evaluate("q", "a", "CENG336")
    assert err.value.status == 400 and "No indexed material" in str(err.value)


def test_unreachable_backend():
    import httpx
    api.use_client(httpx.Client(base_url="http://127.0.0.1:9", timeout=1))
    try:
        with pytest.raises(api.ApiError, match="Cannot reach the backend"):
            api.health()
    finally:
        api.use_client(None)
