"""Tests for llm/, tasks/ and evaluation/ragas_runner without any API key or model:
LLM calls and retrieval are replaced by fakes, so these check the logic around them."""
import json

import pytest

from backend import config
from backend.llm import providers
from backend.llm.providers import parse_json, resolve_provider_name


# --------------------------------------------------------------------------- llm/providers
def test_parse_json_handles_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"claims": [{"supported": true}]} Hope it helps.') == \
        {"claims": [{"supported": True}]}
    with pytest.raises(ValueError):
        parse_json("no json here")


@pytest.mark.parametrize("label,expected", [
    ("Groq (Llama 3)", "groq"), ("OpenAI (GPT-4o-mini)", "openai"),
    ("Ollama (Mistral) — Gizlilik modu", "ollama"), (None, None), ("", None)])
def test_resolve_provider_name(label, expected):
    assert resolve_provider_name(label) == expected


class _Fake:
    def __init__(self, name, fail):
        self.name, self.fail = name, fail

    def chat(self, messages, **kw):
        if self.fail:
            raise RuntimeError(f"{self.name} down")
        return f"answer from {self.name}"


def _providers(monkeypatch, failing):
    calls = []

    def get(name=None):
        calls.append(name)
        return _Fake(name, name in failing)
    monkeypatch.setattr(providers, "get_provider", get)
    return calls


def test_fallback_order(monkeypatch):
    monkeypatch.setitem(config.get_config()["llm"], "fallback", ["openai"])
    calls = _providers(monkeypatch, failing={"groq"})
    text, used = providers.chat_with_fallback([], provider="groq")
    assert (text, used) == ("answer from openai", "openai") and calls == ["groq", "openai"]


def test_privacy_mode_never_falls_back_to_cloud(monkeypatch):
    monkeypatch.setitem(config.get_config()["llm"], "fallback", ["openai", "groq"])
    calls = _providers(monkeypatch, failing={"ollama"})
    with pytest.raises(RuntimeError):
        providers.chat_with_fallback([], provider="ollama")
    assert calls == ["ollama"]


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    providers._CACHE.clear()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        providers.LLMProvider("groq")


# --------------------------------------------------------------------------- ragas_runner
def test_grounding_thresholds():
    from backend.evaluation.ragas_runner import classify
    assert classify(0.9) == "grounded"
    assert classify(0.85) == "grounded"
    assert classify(0.6) == "partially_grounded"
    assert classify(0.49) == "unsupported"


def test_faithfulness_is_share_of_supported_claims(monkeypatch):
    from backend.evaluation import ragas_runner
    reply = json.dumps({"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False},
                                   {"claim": "c", "supported": True}, {"claim": "d", "supported": True}]})
    monkeypatch.setattr(ragas_runner, "chat_with_fallback", lambda *a, **k: (reply, "groq"))
    assert ragas_runner.faithfulness_score("x", ["ctx"]) == 0.75


# --------------------------------------------------------------------------- tasks/question_gen
def test_question_formats_are_validated(monkeypatch):
    from backend.tasks import question_gen as qg
    assert [qg._canonical(t) for t in ["Multiple Choice", "True/False", "Short Answer",
                                       "Conceptual/Computational"]] == ["mcq", "tf", "short", "concept"]

    class Store:
        def get(self, where=None, limit=None):
            return [{"text": "Nyquist", "metadata": {"source_type": "pdf", "document": "k.pdf", "page": 3}}]

    # n=4 with [MC, TF] -> requested plan mcq, tf, mcq, tf (items are checked against this order)
    reply = json.dumps({"questions": [
        {"question": "Q1?", "options": ["a", "b", "c", "d"], "correct_answer": "b", "source_index": 1},
        {"question": "Q2?", "options": None, "correct_answer": "True"},
        {"question": "Q3?", "options": ["a", "b", "c"], "correct_answer": "a"},          # 3 options -> dropped
        {"question": "Q4?", "options": None, "correct_answer": "maybe"},               # not True/False -> dropped
    ]})
    monkeypatch.setattr(qg, "chat_with_fallback", lambda *a, **k: (reply, "groq"))
    out = qg.generate_questions("CENG384", 5, ["Multiple Choice", "True/False"], 4, "Medium", store=Store())
    assert [q["question"] for q in out] == ["Q1?", "Q2?"]
    assert out[0]["source"] == "k.pdf, p. 3" and out[0]["type"] == "Multiple Choice"


def test_question_gen_without_material_is_a_400(monkeypatch):
    from backend.tasks import question_gen as qg

    class Empty:
        def get(self, where=None, limit=None):
            return []
    with pytest.raises(ValueError):
        qg.generate_questions("X", None, ["Short Answer"], 1, "Easy", store=Empty())


# --------------------------------------------------------------------------- tasks/evaluator
def test_concept_coverage_is_fuzzy_and_deterministic():
    from backend.tasks.evaluator import concept_coverage
    answer = "Örnekleme frekansı en yüksek frekansın iki katı olmalı, yoksa aliasing oluşur."
    assert concept_coverage(answer, ["aliasing", "örnekleme frekansı", "Nyquist oranı"]) == pytest.approx(2 / 3)
    assert concept_coverage(answer, []) is None


def test_rubric_total_uses_config_weights(monkeypatch):
    from backend.tasks import evaluator as ev
    chunk = {"text": "ctx", "metadata": {"source_type": "pdf", "document": "k.pdf", "page": 1}}
    monkeypatch.setattr(ev, "retrieve", lambda *a, **k: {"chunks": [chunk]})
    monkeypatch.setattr(ev, "terminology_score", lambda a, c: 0.5)
    reply = json.dumps({"factual_accuracy": 1.0, "completeness": 0.0, "key_concepts": ["aliasing"],
                        "feedback": ["ok"], "follow_up_question": "?"})
    monkeypatch.setattr(ev, "chat_with_fallback", lambda *a, **k: (reply, "groq"))
    r = ev.evaluate_answer("q", "aliasing happens", "CENG384")
    # 0.4*1.0 + 0.3*1.0 (concept found) + 0.2*0.5 + 0.1*0.0 = 0.8
    assert r["total"] == 80 and [d["name"] for d in r["dims"]][0] == "Factual accuracy"


def test_evaluation_without_material_is_refused(monkeypatch):
    from backend.tasks import evaluator as ev
    monkeypatch.setattr(ev, "retrieve", lambda *a, **k: {"chunks": []})
    with pytest.raises(ValueError):
        ev.evaluate_answer("q", "a", "")
