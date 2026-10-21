"""The "beyond the course material" part: general knowledge / reasoning, kept apart from the
grounded answer, never scored by the faithfulness check, and switchable in config.yaml."""
import pytest

from backend import config
from backend.llm.prompts import BEYOND_MARKER, build_qa_messages, build_study_messages, split_beyond

CHUNK = {"text": "Signals vary with time.", "score": 0.6,
         "metadata": {"source_type": "pdf", "document": "k.pdf", "page": 3,
                      "timestamp_start_ms": None, "timestamp_end_ms": None}}


@pytest.mark.parametrize("raw,main,extra", [
    ("DSP uses filters [1].\n\n[BEYOND] Filters also appear in audio apps.", "DSP uses filters [1].",
     "Filters also appear in audio apps."),
    ("A [2].\n**[BEYOND]**: extra", "A [2].", "extra"),
    ("a [1]\n[beyond] b", "a [1]", "b"),
    ("only the material [1]", "only the material [1]", None),
    ("a [1]\n[BEYOND]", "a [1]", None),
])
def test_split_beyond(raw, main, extra):
    assert split_beyond(raw) == (main, extra)


def test_prompts_allow_beyond_only_when_enabled():
    on = build_qa_messages("q", [CHUNK], beyond=True)[0]["content"]
    off = build_qa_messages("q", [CHUNK])[0]["content"]
    assert BEYOND_MARKER in on and "do not answer from general knowledge" not in on
    assert BEYOND_MARKER not in off and "do not answer from general knowledge" in off
    study_on = build_study_messages("k.pdf", "ask", [CHUNK], question="q", beyond=True)[0]["content"]
    study_off = build_study_messages("k.pdf", "ask", [CHUNK], question="q")[0]["content"]
    assert BEYOND_MARKER in study_on and "do not answer from general knowledge" not in study_on
    assert BEYOND_MARKER not in study_off


def _qa(monkeypatch, reply):
    import backend.tasks.qa as qa
    monkeypatch.setattr(qa, "retrieve", lambda *a, **k: {"chunks": [CHUNK], "retrieval_ms": 5,
                                                          "candidates": 1, "filtered_from": 9})
    monkeypatch.setattr(qa, "count_tokens", lambda text: 1)
    monkeypatch.setattr(qa, "chat_with_fallback", lambda messages, **k: (reply, "groq"))
    scored = []
    monkeypatch.setattr(qa, "faithfulness_score", lambda answer, ctx, used: scored.append(answer) or 0.9)
    return qa, scored


def test_faithfulness_scores_only_the_material_part(monkeypatch):
    qa, scored = _qa(monkeypatch, "Signals vary with time [1].\n[BEYOND] Most real signals are analog.")
    r = qa.answer_question("What is a signal?", "C")
    assert r["answer"] == "Signals vary with time [1]." and r["beyond"] == "Most real signals are analog."
    assert r["found"] is True and scored == ["Signals vary with time [1]."]


def test_not_found_with_general_answer(monkeypatch):
    qa, scored = _qa(monkeypatch, "[NO_INFO] The materials do not cover this.\n[BEYOND] The FFT is O(N log N).")
    r = qa.answer_question("FFT complexity?", "C")
    assert r["found"] is False and r["answer"] == "The materials do not cover this."
    assert r["beyond"] == "The FFT is O(N log N)." and scored == [] and r["faithfulness"] == 1.0


def test_switched_off_keeps_the_strict_behaviour(monkeypatch):
    monkeypatch.setitem(config.get_config()["qa"], "general_knowledge", False)
    qa, _ = _qa(monkeypatch, "text [1] [BEYOND] stays as written")
    seen = {}
    monkeypatch.setattr(qa, "build_qa_messages",
                        lambda q, c, beyond=False: seen.update(beyond=beyond) or [{"role": "user", "content": q}])
    r = qa.answer_question("q", "C")
    assert seen["beyond"] is False and r["beyond"] is None and "[BEYOND]" in r["answer"]


def test_study_answer_returns_beyond(monkeypatch):
    import backend.tasks.study as study
    monkeypatch.setattr(study, "prepare", lambda **k: {"messages": [], "sources": [], "mode": "ask",
                                                        "language": "en", "retrieval_ms": 1, "max_tokens": 10})
    monkeypatch.setattr(study, "chat_with_fallback", lambda *a, **k: ("From the doc [1].\n[BEYOND] Extra.", "groq"))
    r = study.study_answer(course="C", document="d.pdf", mode="ask", question="q")
    assert r["answer"] == "From the doc [1]." and r["beyond"] == "Extra."


def test_ui_stream_shows_a_separator_and_keeps_the_raw_text():
    pytest.importorskip("streamlit")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag_ui"))
    from core.state import BeyondStream, split_beyond as ui_split
    pieces = ["From the doc [1].\n[BEY", "OND] General", " idea."]   # marker split over pieces
    stream = BeyondStream(iter(pieces))
    shown = "".join(stream)
    assert "[BEY" not in shown and "Beyond the course material" in shown and shown.endswith("General idea.")
    assert stream.raw == "".join(pieces)
    assert ui_split(stream.raw) == ("From the doc [1].", "General idea.")
