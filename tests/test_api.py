"""HTTP-level tests of backend/main.py (FastAPI TestClient). Tasks are replaced by fakes, so the
tests check routing, request parsing, error codes and the response keys in docs/api_contract.md."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402

client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_documents(monkeypatch):
    class Store:
        def list_documents(self):
            return [{"document": "a.pdf", "course": "C", "lecture": 1, "type": "pdf", "chunks": 3}]
    monkeypatch.setattr(main, "get_store", lambda: Store())
    assert client.get("/documents").json()["documents"][0]["chunks"] == 3


def test_ingest_saves_and_indexes(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "resolve_path", lambda p: tmp_path)
    seen = {}

    def fake_index(path, course, lecture):
        seen.update(path=path, course=course, lecture=lecture)
        return {"document": path.name, "course": course, "lecture": lecture, "type": "pdf", "chunks": 7}
    monkeypatch.setattr(main, "index_file", fake_index)
    r = client.post("/ingest", data={"course": "CENG336", "lecture": "1"},
                    files=[("files", ("../../evil/Chapter_1.pdf", b"%PDF-1.4", "application/pdf"))])
    assert r.status_code == 200 and r.json()["records"][0]["chunks"] == 7
    assert seen["path"] == tmp_path / "CENG336" / "Chapter_1.pdf"      # path components stripped
    assert seen["path"].read_bytes() == b"%PDF-1.4" and seen["lecture"] == 1


def test_ingest_rejects_unsupported_type(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "resolve_path", lambda p: tmp_path)
    r = client.post("/ingest", data={"course": "C", "lecture": "1"},
                    files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_ingest_text_instead_of_file_is_422():
    # what Swagger UI sends when a path is typed into the "files" box
    r = client.post("/ingest", data={"course": "C", "lecture": "1", "files": "C:\\x\\a.pdf"})
    assert r.status_code == 422


def test_ask_converts_minutes_and_hides_contexts(monkeypatch):
    import backend.tasks.qa as qa
    seen = {}

    def fake_answer(question, course, lecture, window, cfg):
        seen.update(window=window, lecture=lecture, cfg=cfg)
        return {"answer": "x [1]", "faithfulness": None, "sources": [], "contexts": ["secret"],
                "retrieval_ms": 1, "total_s": 0.1, "tokens_sent": 10, "candidates": 0, "filtered_from": 0}
    monkeypatch.setattr(qa, "answer_question", fake_answer)
    r = client.post("/ask", json={"question": "q", "course": "C", "lecture": 2, "ts_range": [12, 14]})
    body = r.json()
    assert r.status_code == 200 and "contexts" not in body and body["faithfulness"] is None
    assert seen["window"] == (720_000, 840_000) and seen["lecture"] == 2


@pytest.mark.parametrize("exc,code", [(ValueError("no material"), 400), (RuntimeError("Missing GROQ_API_KEY"), 503)])
def test_error_codes(monkeypatch, exc, code):
    import backend.tasks.evaluator as ev

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(ev, "evaluate_answer", boom)
    r = client.post("/evaluate", json={"question": "q", "answer": "a", "course": "C"})
    assert r.status_code == code and str(exc) in r.json()["detail"]
