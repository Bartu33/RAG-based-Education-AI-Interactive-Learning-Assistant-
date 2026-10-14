"""Ratings and SUS surveys (backend/evaluation/feedback.py + endpoints), written to a temp folder."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402
from backend import config  # noqa: E402
from backend.evaluation import feedback as fb  # noqa: E402
from backend.tasks.qa import split_no_info  # noqa: E402

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def tmp_feedback(monkeypatch, tmp_path):
    monkeypatch.setitem(config.get_config(), "feedback", {"path": str(tmp_path)})


def test_sus_scoring_follows_the_standard():
    assert fb.sus_score([5, 1] * 5) == 100.0       # best possible answers
    assert fb.sus_score([1, 5] * 5) == 0.0         # worst possible answers
    assert fb.sus_score([3] * 10) == 50.0
    assert fb.sus_grade(90) == "Excellent" and fb.sus_grade(68) == "OK" and fb.sus_grade(20) == "Awful"
    with pytest.raises(ValueError):
        fb.sus_score([3] * 9)
    with pytest.raises(ValueError):
        fb.sus_score([0] + [3] * 9)


def test_rating_with_same_id_replaces_the_earlier_one():
    client.post("/feedback", json={"id": "a", "kind": "qa", "up": True})
    client.post("/feedback", json={"id": "a", "kind": "qa", "up": False, "comment": "wrong page"})
    client.post("/feedback", json={"id": "b", "kind": "qa", "up": True})
    client.post("/feedback", json={"id": "c", "kind": "evaluation", "up": True})
    s = client.get("/feedback/summary").json()
    assert s["ratings"]["qa"] == {"count": 2, "up": 1, "down": 1, "positive_pct": 50.0}
    assert s["ratings"]["evaluation"]["count"] == 1
    assert s["comments"][0]["comment"] == "wrong page"


def test_survey_endpoint_scores_and_validates():
    r = client.post("/survey", json={"responses": [4, 2] * 5})
    assert r.status_code == 200 and r.json() == {"score": 75.0, "grade": "Good"}
    assert client.post("/survey", json={"responses": [3] * 4}).status_code == 400
    assert client.post("/feedback", json={"kind": "other", "up": True}).status_code == 400
    s = client.get("/feedback/summary").json()["sus"]
    assert s["count"] == 1 and s["mean"] == 75.0


def test_empty_summary():
    s = client.get("/feedback/summary").json()
    assert s["sus"]["mean"] is None and s["ratings"]["qa"]["positive_pct"] is None


def test_no_info_marker_is_detected_and_removed():
    assert split_no_info("[NO_INFO] The course materials do not cover this.") == \
        ("The course materials do not cover this.", False)
    assert split_no_info("[NO_INFO]") == ("I could not find this in the approved course materials.", False)
    assert split_no_info("DSP is ... [1]") == ("DSP is ... [1]", True)


def test_not_found_answer_skips_the_faithfulness_call(monkeypatch):
    import backend.tasks.qa as qa
    chunk = {"text": "Signals vary with time.", "score": 0.4,
             "metadata": {"source_type": "pdf", "document": "k.pdf", "page": 3,
                          "timestamp_start_ms": None, "timestamp_end_ms": None}}
    monkeypatch.setattr(qa, "retrieve", lambda *a, **k: {"chunks": [chunk], "retrieval_ms": 5,
                                                          "candidates": 1, "filtered_from": 9})
    monkeypatch.setattr(qa, "count_tokens", lambda text: 1)
    monkeypatch.setattr(qa, "chat_with_fallback",
                        lambda *a, **k: ("[NO_INFO] The materials do not cover this.", "groq"))

    def must_not_run(*a, **k):
        raise AssertionError("faithfulness must not be scored for a not-found reply")
    monkeypatch.setattr(qa, "faithfulness_score", must_not_run)
    r = qa.answer_question("Who won the 2018 World Cup?", "C")
    assert r["found"] is False and r["faithfulness"] == 1.0
    assert r["answer"] == "The materials do not cover this." and len(r["sources"]) == 1
