"""Past exams (backend/tasks/exam_style.py): analysis, storage, and questions in the instructor's
style, through the HTTP endpoints and the UI client, with the LLM and the search replaced by fakes."""
import json
import random

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402
from backend import config  # noqa: E402
from backend.tasks import exam_style, question_gen as qg  # noqa: E402
from rag_ui.core import api_client as api  # noqa: E402

ANALYSIS = {"summary": "Mostly calculations; every answer must be justified.",
            "style_points": ["gives a signal and asks for its period", ""],
            "topics": ["periodicity", "convolution"],
            "questions": [{"text": "Is x(n)=cos(0.3πn) periodic? Justify.", "topic": "periodicity", "format": "concept"},
                          {"text": "y(n)=x(n)*h(n) for h(n)=δ(n-2) is a delay. (T/F)", "topic": "convolution",
                           "format": "tf"},
                          {"text": "", "topic": "x"},                                   # empty: dropped
                          {"text": "Which is LTI?", "topic": "systems", "format": "weird"}]}  # unknown -> concept


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setitem(config.get_config(), "database", {"path": str(tmp_path / "app.db")})
    monkeypatch.setattr(exam_style, "chat_with_fallback", lambda *a, **k: (json.dumps(ANALYSIS), "groq"))
    api.use_client(TestClient(main.app))
    yield
    api.use_client(None)


def test_analysis_is_normalised():
    style = exam_style.normalize_style(ANALYSIS)
    assert [q["format"] for q in style["questions"]] == ["concept", "tf", "concept"]
    assert style["mix"] == {"mcq": 0, "tf": 1, "short": 0, "concept": 2}
    assert style["style_points"] == ["gives a signal and asks for its period"]
    assert exam_style.normalize_style({})["questions"] == []


def test_exam_without_questions_is_refused(monkeypatch):
    monkeypatch.setattr(exam_style, "chat_with_fallback", lambda *a, **k: ('{"summary": "?"}', "groq"))
    with pytest.raises(api.ApiError, match="No exam questions"):
        api.add_exam("CENG336", "M", "some text")


def test_add_list_and_remove_exams():
    e = api.add_exam("CENG336", "Midterm 2023", "1) Is x(n) periodic? ...")
    assert e["name"] == "Midterm 2023" and len(e["questions"]) == 3 and not e["truncated"]
    assert [x["id"] for x in api.exams("CENG336")] == [e["id"]] and api.exams("OTHER") == []
    api.delete_exam(e["id"])
    assert api.exams("CENG336") == []
    with pytest.raises(api.ApiError) as err:
        api.delete_exam(e["id"])
    assert err.value.status == 404


def test_exam_file_upload(tmp_path):
    with pytest.raises(api.ApiError, match="PDF, DOCX or PPTX"):
        api.add_exam("CENG336", file=("exam.mp4", b"x", "video/mp4"))
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Q1. Is cos(0.3 pi n) periodic?")
    e = api.add_exam("CENG336", file=("final_2022.pdf", doc.tobytes(), "application/pdf"))
    assert e["name"] == "final_2022"
    blank = fitz.open()
    blank.new_page()
    with pytest.raises(api.ApiError, match="No text found"):
        api.add_exam("CENG336", file=("scan.pdf", blank.tobytes(), "application/pdf"))


def test_examples_and_topics_come_from_every_exam():
    a = {"name": "A", "topics": ["t1"], "questions": [{"text": f"a{i}", "topic": "t1", "format": "tf"} for i in range(5)]}
    b = {"name": "B", "topics": [], "questions": [{"text": "b0", "topic": "t2", "format": "mcq"}]}
    ex = exam_style.pick_examples([a, b], k=3, rng=random.Random(0))
    assert len(ex) == 3 and {e["exam"] for e in ex} == {"A", "B"}
    assert set(exam_style.exam_topics([a, b], rng=random.Random(0))) == {"t1", "t2"}


def test_generation_imitates_the_chosen_exams(monkeypatch):
    import backend.retrieval.retriever as retriever
    exam = api.add_exam("CENG336", "Midterm 2023", "exam text")
    searched = []

    def fake_retrieve(q, course, lecture=None, **kw):
        searched.append(q)
        return {"chunks": [{"text": f"about {q}", "metadata": {"source_type": "pdf", "document": "c.pdf", "page": 4}}]}
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    prompts = []
    reply = json.dumps({"questions": [{"format": "concept", "question": "Is cos(0.5πn) periodic? Justify.",
                                       "correct_answer": "yes, N=4", "source_index": 1, "similar_to": 1}]})
    monkeypatch.setattr(qg, "chat_with_fallback", lambda m, **k: prompts.append(m[1]["content"]) or (reply, "groq"))

    out = api.generate_questions("CENG336", None, {"Conceptual/Computational": 1}, "Medium",
                                 exam_ids=[exam["id"]])
    assert set(searched) <= {"periodicity", "convolution", "systems"} and searched   # the exams' topics
    assert "INSTRUCTOR'S STYLE" in prompts[0] and "E1. [" in prompts[0] and "every answer must be justified" in prompts[0]
    assert out[0]["style_ref"].startswith("Midterm 2023: ") and out[0]["source"] == "c.pdf, p. 4"

    with pytest.raises(api.ApiError, match="do not belong"):
        api.generate_questions("OTHER", None, {"Short Answer": 1}, "Medium", exam_ids=[exam["id"]])
