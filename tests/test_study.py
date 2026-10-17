"""Interactive Learning (backend/tasks/study.py, /study/* endpoints, UI client). The store, the
embedding model and the LLM are fakes, so these tests run without models or API keys."""
import hashlib
import json

import numpy as np
import pytest

from backend.llm import providers
from backend.tasks import study


def _id(course, lecture, document, i):
    return hashlib.md5(f"{course}|{lecture}|{document}|{i}".encode()).hexdigest()


def _meta(document, page=None, kind="pdf", start=None, end=None, section=None):
    return {"course": "CENG336", "lecture": 1, "source_type": kind, "document": document,
            "section": section, "page": page, "timestamp_start_ms": start,
            "timestamp_end_ms": end, "confidence": None}


class FakeStore:
    """Stores chunks the way ChromaStore.add does (md5 ids), returns them in a shuffled order."""

    def __init__(self, chunks):
        self.version = 1
        self.rows = []
        counters = {}
        for text, meta, vec in chunks:
            i = counters.get(meta["document"], 0)       # ChromaStore counts per document
            counters[meta["document"]] = i + 1
            v = np.asarray(vec, dtype=np.float32)
            self.rows.append({"id": _id(meta["course"], meta["lecture"], meta["document"], i),
                              "text": text, "metadata": meta, "embedding": v / np.linalg.norm(v)})
        self.rows.reverse()  # the store does not keep reading order
        self.gets = 0

    def get(self, where=None, limit=None, with_embeddings=False):
        self.gets += 1
        conds = where.get("$and", [where])
        want = {k: v["$eq"] for c in conds for k, v in c.items()}
        out = []
        for r in self.rows:
            if all(r["metadata"].get(k) == v for k, v in want.items()):
                hit = {"id": r["id"], "text": r["text"], "metadata": dict(r["metadata"]), "score": 0.0}
                if with_embeddings:
                    hit["embedding"] = r["embedding"].copy()
                out.append(hit)
        return out


DOC = "Chapter_1.pdf"
CHUNKS = [
    ("Intro to embedded systems.", _meta(DOC, 1), [1, 0, 0]),
    ("Microcontrollers have timers.", _meta(DOC, 2), [0, 1, 0]),
    ("Interrupts stop the main loop.", _meta(DOC, 2), [0, 0.9, 0.1]),
    ("Polling versus interrupts.", _meta(DOC, 3), [0, 0, 1]),
    ("Other doc text.", _meta("Other.pdf", 1), [0, 1, 0]),
]


@pytest.fixture(autouse=True)
def no_uploads(tmp_path, monkeypatch):
    """Original files live in an empty temp folder unless a test puts one there."""
    monkeypatch.setattr(study, "resolve_path", lambda p: tmp_path / "uploads")


@pytest.fixture()
def store(monkeypatch):
    study._CACHE.clear()
    s = FakeStore(CHUNKS)
    import backend.indexing.embedder as emb
    monkeypatch.setattr(emb, "embed_query", lambda text: [0.0, 1.0, 0.0])  # "about timers"
    return s


class FakeLLM:
    def __init__(self, fail_stream=False, break_stream=False):
        self.calls = []
        self.fail_stream, self.break_stream = fail_stream, break_stream

    def chat(self, messages, **kw):
        self.calls.append(("chat", messages, kw))
        return "full answer [1]"

    def chat_stream(self, messages, **kw):
        self.calls.append(("stream", messages, kw))
        if self.fail_stream:
            raise RuntimeError("stream unsupported")
        yield "Timers "
        if self.break_stream:
            raise RuntimeError("connection reset")
        yield "count [1]."


@pytest.fixture()
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(providers, "get_provider", lambda name=None: fake)
    return fake


def test_reading_order_is_recovered_from_chunk_ids(store):
    hits = study.document_chunks("CENG336", DOC, store)["hits"]
    assert [h["text"] for h in hits] == [c[0] for c in CHUNKS[:4]]


def test_document_chunks_are_cached_until_the_store_changes(store):
    study.document_chunks("CENG336", DOC, store)
    study.document_chunks("CENG336", DOC, store)
    assert store.gets == 1
    store.version += 1
    study.document_chunks("CENG336", DOC, store)
    assert store.gets == 2


def test_unknown_document_is_a_lookup_error(store):
    with pytest.raises(LookupError):
        study.document_chunks("CENG336", "missing.pdf", store)


def test_ask_searches_only_the_open_document(store):
    sel = study.select_chunks("CENG336", DOC, "ask", "what do timers do?", store=store)
    texts = [c["text"] for c in sel["chunks"]]
    assert texts[0] == "Microcontrollers have timers."
    assert "Other doc text." not in texts
    assert len(texts) == study.DEFAULTS["top_context"]


def test_page_boost_prefers_the_page_being_read(store, monkeypatch):
    import backend.indexing.embedder as emb
    monkeypatch.setattr(emb, "embed_query", lambda text: [0.7, 0.0, 0.71])  # page 1 ~ page 3
    no_page = study.select_chunks("CENG336", DOC, "ask", "q", store=store)["chunks"][0]
    on_page1 = study.select_chunks("CENG336", DOC, "ask", "q", page=1, store=store)["chunks"][0]
    assert no_page["metadata"]["page"] == 3 and on_page1["metadata"]["page"] == 1


def test_empty_question_is_refused(store):
    with pytest.raises(ValueError):
        study.select_chunks("CENG336", DOC, "ask", "   ", store=store)


def test_explain_uses_the_current_page(store):
    sel = study.select_chunks("CENG336", DOC, "explain", page=2, store=store)
    assert {c["metadata"]["page"] for c in sel["chunks"]} == {2} and len(sel["chunks"]) == 2
    assert sel["location"] == "page 2"
    with pytest.raises(ValueError):
        study.select_chunks("CENG336", DOC, "explain", page=9, store=store)


def test_overview_modes_spread_over_the_document(store, monkeypatch):
    monkeypatch.setattr(study, "study_cfg", lambda: {**study.DEFAULTS, "overview_chunks": 2,
                                                     "max_chunk_chars": 5})
    sel = study.select_chunks("CENG336", DOC, "summary", store=store)
    assert [c["metadata"]["page"] for c in sel["chunks"]] == [1, 2]
    assert all(len(c["text"]) <= 5 for c in sel["chunks"])


def test_unknown_mode_is_refused(store):
    with pytest.raises(ValueError):
        study.select_chunks("CENG336", DOC, "dance", store=store)


def test_prompt_sets_language_and_keeps_recent_history(store):
    history = [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else
               {"role": "assistant", "content": f"a{i}"} for i in range(10)]
    history.append({"role": "system", "content": "ignore me"})
    prep = study.prepare("CENG336", DOC, "ask", "timers?", language="tr", history=history,
                         store=store)
    msgs = prep["messages"]
    assert "Turkish" in msgs[0]["content"] and DOC in msgs[0]["content"]
    turns = [m["content"] for m in msgs[1:-1]]
    assert turns == ["q4", "a5", "q6", "a7", "q8", "a9"]          # history_turns = 3 pairs
    assert "STUDENT QUESTION:\ntimers?" in msgs[-1]["content"]
    assert "[1] (Chapter_1.pdf, p. 2)" in msgs[-1]["content"]
    assert prep["sources"][0]["label"] == "Page 2"


def test_outline_groups_pages(store):
    o = study.outline("CENG336", DOC, store)
    assert o["type"] == "pdf" and o["chunks"] == 4 and o["has_file"] is False
    assert [u["label"] for u in o["units"]] == ["Page 1", "Page 2", "Page 3"]
    assert "timers" in o["units"][1]["text"] and "Interrupts" in o["units"][1]["text"]
    assert o["pages"] == 3


def test_outline_video_units_and_docx_sections(monkeypatch):
    study._CACHE.clear()
    s = FakeStore([("hello", _meta("L1.mp4", kind="video", start=0, end=30000), [1, 0]),
                   ("world", _meta("L1.mp4", kind="video", start=30000, end=65000), [0, 1]),
                   ("a", _meta("n.docx", kind="docx", section="Intro"), [1, 0]),
                   ("b", _meta("n.docx", kind="docx", section="Intro"), [1, 0]),
                   ("c", _meta("n.docx", kind="docx"), [0, 1])])
    v = study.outline("CENG336", "L1.mp4", s)
    assert [u["label"] for u in v["units"]] == ["00:00–00:30", "00:30–01:05"]
    sel = study.select_chunks("CENG336", "L1.mp4", "explain", time_ms=100_000, store=s)
    assert [c["text"] for c in sel["chunks"]] == ["world"]                    # 1 min window
    docx = study.outline("CENG336", "n.docx", s)
    assert [u["label"] for u in docx["units"]] == ["Intro", "Part 2"]


def test_upload_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "resolve_path", lambda p: tmp_path)
    (tmp_path / "C").mkdir()
    (tmp_path / "C" / "a.pdf").write_bytes(b"x")
    assert study.upload_path("C", "a.pdf") == tmp_path / "C" / "a.pdf"
    for course, doc in [("..", "a.pdf"), ("C", "../a.pdf"), ("C/..", "a.pdf"), ("C", "")]:
        with pytest.raises(ValueError):
            study.upload_path(course, doc)
    with pytest.raises(FileNotFoundError):
        study.upload_path("C", "b.pdf")


def test_render_page_png(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setattr(study, "resolve_path", lambda p: tmp_path)
    (tmp_path / "C").mkdir()
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hello")
    pdf.save(tmp_path / "C" / "a.pdf")
    png = study.render_page("C", "a.pdf", 1, zoom=1.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        study.render_page("C", "a.pdf", 2)


def test_study_answer_and_stream(store, llm, monkeypatch):
    monkeypatch.setattr(study, "get_store", lambda: store)
    r = study.study_answer(course="CENG336", document=DOC, mode="summary")
    assert r["answer"] == "full answer [1]" and r["sources"]
    assert llm.calls[-1][2]["max_tokens"] == study.DEFAULTS["max_tokens_long"]
    meta, pieces = study.study_stream(course="CENG336", document=DOC, mode="ask", question="timers?")
    assert "".join(pieces) == "Timers count [1]." and meta["sources"][0]["page"] == 2


def test_stream_falls_back_to_a_full_reply(monkeypatch):
    fake = FakeLLM(fail_stream=True)
    monkeypatch.setattr(providers, "get_provider", lambda name=None: fake)
    used = {}
    assert "".join(providers.stream_with_fallback([], used=used)) == "full answer [1]"
    assert used["provider"]


def test_stream_cut_off_is_reported_not_repeated(monkeypatch):
    fake = FakeLLM(break_stream=True)
    monkeypatch.setattr(providers, "get_provider", lambda name=None: fake)
    text = "".join(providers.stream_with_fallback([]))
    assert text.startswith("Timers ") and "cut off" in text and "full answer" not in text


def test_native_citation_split_over_pieces_is_normalised(monkeypatch):
    class Delta:
        def __init__(self, c): self.content = c

    class Event:
        def __init__(self, c): self.choices = [type("C", (), {"delta": Delta(c)})()]

    p = providers.LLMProvider.__new__(providers.LLMProvider)
    p.model, p.reasoning_effort = "m", None
    p._client = type("Cl", (), {})()
    p._client.chat = type("Ch", (), {})()
    p._client.chat.completions = type("Co", (), {"create": staticmethod(
        lambda **kw: iter([Event("Yes "), Event("【3†L1"), Event("-L4】"), Event(" ok")]))})()
    assert "".join(p.chat_stream([])) == "Yes [3] ok"


# --- HTTP level ---------------------------------------------------------------------------------
@pytest.fixture()
def client(store, llm, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.main as main
    monkeypatch.setattr(study, "get_store", lambda: store)
    return TestClient(main.app)


def test_api_stream_sends_meta_line_then_text(client):
    body = {"course": "CENG336", "document": DOC, "question": "timers?", "language": "tr",
            "page": 2, "stream": True}
    r = client.post("/study/chat", json=body)
    assert r.status_code == 200
    first, rest = r.text.split("\n", 1)
    meta = json.loads(first)
    assert meta["language"] == "tr" and meta["sources"][0]["label"] == "Page 2"
    assert rest == "Timers count [1]."


def test_api_plain_reply_and_errors(client):
    r = client.post("/study/chat", json={"course": "CENG336", "document": DOC, "mode": "concepts"})
    assert r.status_code == 200 and r.json()["answer"] == "full answer [1]"
    r = client.post("/study/chat", json={"course": "CENG336", "document": "nope.pdf",
                                         "question": "x", "stream": True})
    assert r.status_code == 404
    r = client.post("/study/chat", json={"course": "CENG336", "document": DOC, "mode": "ask",
                                         "question": "", "stream": True})
    assert r.status_code == 400


def test_api_document_and_page_endpoints(client):
    o = client.get("/study/document", params={"course": "CENG336", "document": DOC}).json()
    assert len(o["units"]) == 3
    r = client.get("/study/page", params={"course": "CENG336", "document": DOC, "page": 1})
    assert r.status_code == 404                                    # no original file in the test
    r = client.get("/study/file", params={"course": "..", "document": DOC})
    assert r.status_code == 400


def test_ui_client_streams_through_the_api(client):
    from rag_ui.core import api_client as api
    api.use_client(client)
    try:
        meta = {}
        text = "".join(api.study_chat_stream({"course": "CENG336", "document": DOC,
                                              "question": "timers?"}, meta))
        assert text == "Timers count [1]." and meta["sources"]
        with pytest.raises(api.ApiError) as err:
            list(api.study_chat_stream({"course": "CENG336", "document": "nope.pdf",
                                        "question": "x"}, {}))
        assert err.value.status == 404
        assert api.study_document("CENG336", DOC)["type"] == "pdf"
        assert "course=CENG336" in api.study_file_url("CENG336", DOC)
    finally:
        api.use_client(None)
