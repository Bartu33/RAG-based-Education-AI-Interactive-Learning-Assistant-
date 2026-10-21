"""Teaching board (backend/tasks/board.py): no model, no LLM, real PyMuPDF on a generated PDF."""
import hashlib
import json

import numpy as np
import pytest

fitz = pytest.importorskip("fitz")

from backend.tasks import board  # noqa: E402

DIM = 32


def fake_vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in text.lower().split():
        v[int(hashlib.md5(w.strip(".,?:").encode()).hexdigest(), 16) % DIM] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture()
def uploads(tmp_path, monkeypatch):
    """data/uploads/CENG336/ with a 2-page PDF and a dummy video."""
    import backend.tasks.study as study
    monkeypatch.setattr(study, "resolve_path", lambda p: tmp_path)
    folder = tmp_path / "CENG336"
    folder.mkdir()
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Chapter 1 roadmap")
    p2 = doc.new_page()
    p2.insert_text((72, 100), "Packet switching versus circuit switching")
    p2.insert_text((72, 130), "Circuit switching reserves resources for a call")
    p2.insert_text((72, 160), "Unrelated footer text here")
    doc.save(folder / "Chapter_1.pdf")
    (folder / "Lecture01.mp4").write_bytes(b"\x00" * 16)
    monkeypatch.setattr("backend.indexing.embedder.embed_texts",
                        lambda texts, **k: np.stack([fake_vec(t) for t in texts]))
    monkeypatch.setattr("backend.indexing.embedder.embed_query", lambda t: fake_vec(t).tolist())
    return folder


SOURCES = [
    {"type": "video", "document": "Lecture01.mp4", "page": None, "start_ms": 740_000,
     "end_ms": 797_000, "score": 0.9, "text": "packet switching versus circuit switching resources"},
    {"type": "pdf", "document": "Chapter_1.pdf", "page": 1, "start_ms": None, "end_ms": None,
     "score": 0.5, "text": "Chapter 1 roadmap"},
]


class Store:
    """Answers the video -> page match with page 2 of the PDF."""
    def __init__(self, score=0.8):
        self.score, self.where = score, None

    def query(self, embedding, n, where=None):
        self.where = where
        return [{"id": "x", "text": "Packet switching versus circuit switching", "score": self.score,
                 "metadata": {"document": "Chapter_1.pdf", "source_type": "pdf", "page": 2,
                              "course": "CENG336", "lecture": 1}}]


@pytest.mark.parametrize("q,expected", [
    ("2^10 kaç eder?", True), ("Bu algoritmanın karmaşıklığını hesapla", True),
    ("Solve x + 3 = 5", True), ("Derive the recurrence of merge sort", True),
    ("1000 bitlik paket 100 bps hatta ne kadar sürede iletilir?", True),
    ("5 kg kütleye 10 N kuvvet uygulanırsa ivme nedir?", True),
    ("A 1 Gb/s link shared by users of 100 Mbps, how many users?", True),
    ("Paket anahtarlama nedir?", False), ("What is a protocol?", False),
    ("Olasılık nedir?", False), ("What is the time complexity of binary search?", False),
    ("2 sınıfta kaç öğrenci var?", False)])
def test_whiteboard_only_for_working_out(q, expected):
    assert board.is_math_question(q) is expected


def test_formula_in_answer_alone_does_not_open_whiteboard():
    assert not board.is_math_question("Explain this", "The rate is $\\frac{L}{R}$")


def test_question_language():
    assert board.question_language("Paket anahtarlama nedir?") == "tr"
    assert board.question_language("What is a protocol?") == "en"


def test_pick_sources():
    assert board.pick_video(SOURCES)["start_ms"] == 740_000
    assert board.pick_page_source(SOURCES)["page"] == 1
    assert board.pick_video(SOURCES[1:]) is None


def test_video_matches_slide_of_same_lecture(uploads):
    store = Store()
    page = board.match_page_for_video(board.pick_video(SOURCES), "CENG336", 1, store)
    assert page["page"] == 2 and page["reason"] == "video-match"
    assert {"source_type": {"$in": ["pdf", "pptx"]}} in store.where["$and"]
    assert board.match_page_for_video(board.pick_video(SOURCES), "CENG336", 1, Store(0.1)) is None


def test_highlights_pick_answering_lines(uploads):
    lines = board.page_lines(uploads / "Chapter_1.pdf", 2)
    assert "Unrelated footer text here" in lines
    hl = board.choose_highlights(lines, "packet switching versus circuit switching")
    assert hl and hl[0] == "Packet switching versus circuit switching"
    assert "Unrelated footer text here" not in hl


def test_render_page_draws_highlight(uploads):
    plain = board.render_page("CENG336", "Chapter_1.pdf", 2, [])
    marked = board.render_page("CENG336", "Chapter_1.pdf", 2, ["Packet switching versus circuit switching"])
    assert plain.startswith(b"\x89PNG") and marked.startswith(b"\x89PNG") and plain != marked
    with pytest.raises(ValueError):
        board.render_page("CENG336", "Chapter_1.pdf", 9, [])


def test_pptx_without_libreoffice_is_explained(uploads, monkeypatch):
    (uploads / "Slides.pptx").write_bytes(b"pk")
    monkeypatch.setattr(board, "_soffice", lambda: None)
    with pytest.raises(FileNotFoundError, match="LibreOffice"):
        board.pdf_for("CENG336", "Slides.pptx")


def test_board_for_video_math_question(uploads, monkeypatch):
    reply = json.dumps({"title": "Transmission delay",
                        "steps": [{"write": "$d = L/R$", "say": "First, the formula."},
                                  {"write": "$d = 1000/100 = 10$ s", "say": "Now we plug in."}],
                        "result": "10 s", "closing": "So it takes ten seconds."})
    monkeypatch.setattr(board, "chat_with_fallback", lambda *a, **k: (reply, "groq"))
    b = board.build_board("Calculate the delay for L=1000, R=100", "...", "CENG336", 1, SOURCES,
                          store=Store())
    assert b["view"] == "whiteboard" and b["is_math"] and b["lesson"] is None
    assert b["whiteboard"]["steps"][0] == {"write": "$d = L/R$", "say": "First, the formula."}
    assert b["whiteboard"]["result"] == "10 s" and b["whiteboard"]["closing"]
    assert b["video"] == {"document": "Lecture01.mp4", "start_ms": 740_000, "end_ms": 797_000}
    assert b["page"]["page"] == 2 and b["page"]["reason"] == "video-match" and b["page"]["renderable"]


def test_concept_question_gets_a_lecturer_walkthrough(uploads, monkeypatch):
    calls = []

    def fake_llm(messages, **k):
        calls.append(messages)
        return json.dumps({"beats": [{"line": 1, "say": "Look at the title of this slide."},
                                     {"line": 2, "say": "As I said in the video, a circuit reserves resources."},
                                     {"line": 99, "say": "This beat points at no real line."}],
                           "closing": "That is the difference."}), "groq"
    monkeypatch.setattr(board, "chat_with_fallback", fake_llm)
    b = board.build_board("packet switching versus circuit switching", "Packet switching shares...",
                          "CENG336", 1, SOURCES, store=Store())
    assert b["view"] == "page" and not b["is_math"] and b["whiteboard"] is None
    beats = b["lesson"]["beats"]
    assert beats[0]["line"] == "Packet switching versus circuit switching"
    assert beats[1]["line"] == "Circuit switching reserves resources for a call"
    assert beats[2]["line"] is None and b["lesson"]["narrated"]
    assert b["page"]["highlights"] == [beats[0]["line"], beats[1]["line"]]   # what was pointed at
    prompt = calls[0][1]["content"]
    assert "1. Packet switching versus circuit switching" in prompt and "VIDEO TRANSCRIPT at 12:20" in prompt
    assert len(calls) == 1                                                  # one LLM call, no whiteboard


def test_walkthrough_without_llm_falls_back(uploads, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Missing GROQ_API_KEY")
    monkeypatch.setattr(board, "chat_with_fallback", boom)
    b = board.build_board("Paket anahtarlama ile devre anahtarlama arasındaki fark nedir? packet switching "
                          "versus circuit switching", "", "CENG336", 1, SOURCES, store=Store())
    assert b["view"] == "page" and not b["lesson"]["narrated"]
    assert b["lesson"]["beats"][0]["say"].startswith("Bu konuyu videoda 12:20")      # Turkish fallback
    assert any("explanation unavailable" in n for n in b["notes"])


def test_board_degrades_without_files(uploads, monkeypatch):
    (uploads / "Lecture01.mp4").unlink()
    (uploads / "Chapter_1.pdf").unlink()
    b = board.build_board("What is a protocol?", "", "CENG336", 1, SOURCES, store=Store())
    assert b["video"] is None and b["view"] == "none"
    assert b["page"]["renderable"] is False and len(b["notes"]) == 2


def test_whiteboard_llm_failure_falls_back_to_page(uploads, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Missing GROQ_API_KEY")
    monkeypatch.setattr(board, "chat_with_fallback", boom)
    b = board.build_board("Solve x + 3 = 5", "", "CENG336", 1, SOURCES, store=Store())
    assert b["view"] == "page" and any("Whiteboard unavailable" in n for n in b["notes"])
    assert b["lesson"]["beats"]   # still explained, with the plain walk-through


def test_disabled_board(monkeypatch):
    monkeypatch.setattr(board, "board_cfg", lambda: {**board.DEFAULTS, "enabled": False})
    assert board.build_board("q", "", "C", 1, SOURCES)["view"] == "none"


def test_api_endpoints(uploads, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main
    client = TestClient(main.app)
    monkeypatch.setattr(board, "build_board", lambda *a, **k: {"view": "page", "args": a[:4]})
    r = client.post("/board", json={"question": "q", "answer": "a", "course": "CENG336", "lecture": 1,
                                    "sources": SOURCES})
    assert r.status_code == 200 and r.json()["args"] == ["q", "a", "CENG336", 1]
    png = client.get("/board/page", params=[("course", "CENG336"), ("document", "Chapter_1.pdf"),
                                            ("page", 2), ("hl", "Packet switching versus circuit switching")])
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    missing = client.get("/board/page", params={"course": "CENG336", "document": "nope.pdf", "page": 1})
    assert missing.status_code == 404


def test_lesson_walks_through_several_cited_pages(uploads, monkeypatch):
    sources = [{"type": "pdf", "document": "Chapter_1.pdf", "page": 1, "start_ms": None, "end_ms": None,
                "score": 0.6, "text": "Chapter 1 roadmap"},
               {"type": "pdf", "document": "Chapter_1.pdf", "page": 2, "start_ms": None, "end_ms": None,
                "score": 0.5, "text": "Packet switching versus circuit switching"}]
    prompts = []

    def fake_llm(messages, **k):
        prompts.append(messages[1]["content"])
        return json.dumps({"beats": [{"line": 1, "say": "This is the roadmap.", "point": "Roadmap"},
                                     {"line": "3", "say": "Now the next page.", "point": "Circuits reserve"}],
                           "closing": "Done.", "summary": ["a", "b"]}), "groq"
    monkeypatch.setattr(board, "chat_with_fallback", fake_llm)
    b = board.build_board("packet switching versus circuit switching", "", "CENG336", None, sources)
    assert "PAGE 1 (Chapter_1.pdf, page 1)" in prompts[0] and "PAGE 2 (Chapter_1.pdf, page 2)" in prompts[0]
    assert [p["page"] for p in b["pages"]] == [1, 2] and b["page"]["page"] == 1
    beats = b["lesson"]["beats"]
    assert beats[0] == {"line": "Chapter 1 roadmap", "page": 0, "say": "This is the roadmap.", "point": "Roadmap"}
    assert beats[1]["page"] == 1 and beats[1]["line"] == "Circuit switching reserves resources for a call"
    assert b["pages"][1]["highlights"] == [beats[1]["line"]] and b["lesson"]["summary"] == ["a", "b"]


def test_whiteboard_formulas_are_delimited(uploads, monkeypatch):
    reply = json.dumps({"title": "Product", "given": ["x1(n)=\\{1,-3\\}"], "find": "y(n)",
                        "steps": [{"write": "x1(n)\\,x2(n)=\\{1\\times2\\}", "say": "Multiply."}],
                        "result": "\\(y(n)=\\{2\\}\\)", "check": "Same length.", "closing": "Done."})
    monkeypatch.setattr(board, "chat_with_fallback", lambda *a, **k: (reply, "groq"))
    wb = board.whiteboard("Calculate the product", SOURCES)
    assert wb["steps"][0]["write"] == "$x1(n)\\,x2(n)=\\{1\\times2\\}$"
    assert wb["given"] == ["$x1(n)=\\{1,-3\\}$"] and wb["result"] == "$y(n)=\\{2\\}$"
    assert wb["check"] == "Same length." and wb["find"] == "y(n)"


def test_turkish_question_gets_a_turkish_lesson(uploads, monkeypatch):
    replies = [json.dumps({"beats": [{"line": 1, "say": "Look at this line, it is the key.", "point": "Key"}],
                           "closing": "That is all."}),
               json.dumps({"beats": [{"line": 1, "say": "Şimdi bu satıra bakalım, burada önemli bir nokta var.",
                                      "point": "Önemli"}], "closing": "Bu kadar."})]
    calls = []

    def fake_llm(messages, **k):
        calls.append(messages)
        return replies[len(calls) - 1], "groq"
    monkeypatch.setattr(board, "chat_with_fallback", fake_llm)
    b = board.build_board("Paket anahtarlama ile devre anahtarlama farkı nedir? packet switching", "",
                          "CENG336", 1, SOURCES, store=Store())
    assert "LANGUAGE: the student asked in Turkish" in calls[0][1]["content"]
    assert len(calls) == 2 and "Rewrite" in calls[1][-1]["content"]           # English reply -> rewritten
    assert b["lesson"]["beats"][0]["say"].startswith("Şimdi") and b["language"] == "tr"
