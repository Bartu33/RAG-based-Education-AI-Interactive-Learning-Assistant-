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
    ("Groq (GPT-OSS 120B)", "groq"), ("OpenAI (GPT-4o-mini)", "openai"),
    ("Ollama (Mistral) — Gizlilik modu", "ollama"), (None, None), ("", None)])
def test_resolve_provider_name(label, expected):
    assert resolve_provider_name(label) == expected


def test_native_citations_become_brackets():
    text = "DSP is ... filtering 【3†L1-L4】. Cheaper【4†L6-L7】 and [2] stays."
    assert providers.normalize_citations(text) == "DSP is ... filtering [3]. Cheaper[4] and [2] stays."


def _capture_create(monkeypatch, name):
    """Real LLMProvider with the HTTP call replaced; returns the kwargs sent to the API."""
    monkeypatch.setenv(f"{name.upper()}_API_KEY", "test-key")
    p = providers.LLMProvider(name)
    sent = {}

    class Msg:
        content = "ok"

    class Resp:
        choices = [type("C", (), {"message": Msg})]

    def create(**kw):
        sent.update(kw)
        return Resp
    monkeypatch.setattr(p._client.chat.completions, "create", create)
    return p, sent


def test_reasoning_model_gets_effort_and_token_headroom(monkeypatch):
    cfg = config.get_config()["llm"]
    monkeypatch.setitem(cfg, "reasoning_effort", {"groq": "low"})
    monkeypatch.setitem(cfg, "reasoning_extra_tokens", 1000)
    p, sent = _capture_create(monkeypatch, "groq")
    assert p.chat([], max_tokens=60) == "ok"
    assert sent["reasoning_effort"] == "low" and sent["max_tokens"] == 1060


def test_non_reasoning_provider_unchanged(monkeypatch):
    monkeypatch.setitem(config.get_config()["llm"], "reasoning_effort", {"groq": "low"})
    p, sent = _capture_create(monkeypatch, "openai")
    p.chat([], max_tokens=60)
    assert "reasoning_effort" not in sent and sent["max_tokens"] == 60


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


def _one_chunk_store():
    class Store:
        def get(self, where=None, limit=None):
            return [{"text": f"part {i}", "metadata": {"source_type": "pdf", "document": "k.pdf", "page": i}}
                    for i in range(1, 30)]
    return Store()


def test_counts_make_a_grouped_plan_and_validate_limits():
    from backend.tasks import question_gen as qg
    plan = qg.make_plan({"True/False": 1, "Multiple Choice": 3, "Short Answer": 0})
    assert plan == [("mcq", "Multiple Choice")] * 3 + [("tf", "True/False")]
    assert len(qg.make_plan(types=["Multiple Choice", "Short Answer"], n=3)) == 3   # older form
    for bad in ({"Multiple Choice": 0}, {"Multiple Choice": 11}, {"Multiple Choice": 10, "Short Answer": 10,
                                                                    "True/False": 1}):
        with pytest.raises(ValueError):
            qg.make_plan(bad)


def test_items_are_checked_by_their_own_format_and_capped(monkeypatch):
    from backend.tasks import question_gen as qg
    mcq = lambda i: {"format": "mcq", "question": f"M{i}?", "options": ["a", "b", "c", "d"],  # noqa: E731
                     "correct_answer": "a", "source_index": 2}
    reply = json.dumps({"questions": [
        {"format": "tf", "question": "T1?", "options": None, "correct_answer": "False"},   # reordered: kept
        mcq(1), mcq(2),
        {"format": "short", "question": "S1?", "correct_answer": "x"},                     # not requested
        mcq(3), mcq(4)]})                                                                   # one MCQ too many
    seen = {}

    def chat(messages, **kw):
        seen["prompt"], seen["max_tokens"] = messages[1]["content"], kw["max_tokens"]
        return reply, "groq"
    monkeypatch.setattr(qg, "chat_with_fallback", chat)
    out = qg.generate_questions("C", None, counts={"Multiple Choice": 3, "True/False": 1},
                                difficulty="Hard", store=_one_chunk_store())
    assert [q["question"] for q in out] == ["M1?", "M2?", "M3?", "T1?"]               # grouped, capped
    assert out[-1]["type"] == "True/False" and out[-1]["options"] is None
    assert out[0]["source"].startswith("k.pdf, p. ") and seen["max_tokens"] == 500 + 250 * 4  # random sample
    assert "Write exactly 4 questions" in seen["prompt"] and "Focus topic" not in seen["prompt"]


def test_concept_questions_carry_kind_and_final_answer(monkeypatch):
    from backend.tasks import question_gen as qg
    reply = json.dumps({"questions": [
        {"format": "concept", "kind": "calculation", "question": "fs = 8 kHz. Nyquist frequency?",
         "correct_answer": "f_N = fs / 2 = 4 kHz", "final_answer": " 4 kHz "},
        {"format": "concept", "kind": "conceptual", "question": "Why sample above 2B?",
         "correct_answer": "To avoid aliasing.", "final_answer": "ignored"},       # conceptual -> no result
        {"format": "concept", "question": "Q3?", "correct_answer": "x"},            # no kind -> conceptual
        {"format": "mcq", "question": "M?", "options": ["a", "b", "c", "d"], "correct_answer": "a",
         "kind": "calculation", "final_answer": "5"}]})                             # not concept -> None
    seen = {}

    def chat(messages, **kw):
        seen["system"], seen["max_tokens"] = messages[0]["content"], kw["max_tokens"]
        return reply, "groq"
    monkeypatch.setattr(qg, "chat_with_fallback", chat)
    out = qg.generate_questions("C", None, counts={"Multiple Choice": 1, "Conceptual/Computational": 3},
                                difficulty="Medium", store=_one_chunk_store())
    got = {q["question"]: (q["kind"], q["final_answer"]) for q in out}
    assert got == {"M?": (None, None), "fs = 8 kHz. Nyquist frequency?": ("calculation", "4 kHz"),
                   "Why sample above 2B?": ("conceptual", None), "Q3?": ("conceptual", None)}
    assert seen["max_tokens"] == 500 + 250 * 4 + qg.CONCEPT_EXTRA_TOKENS * 3
    assert "final_answer" in seen["system"]


def test_concept_spec_asks_for_solvable_calculations():
    from backend.llm.prompts import FORMAT_SPECS
    spec = FORMAT_SPECS["concept"]
    assert "calculation" in spec and "final_answer" in spec and "every value needed" in spec


def test_topic_uses_the_search_instead_of_a_random_sample(monkeypatch):
    from backend.tasks import question_gen as qg
    import backend.retrieval.retriever as retriever
    hit = {"text": "sampling", "metadata": {"source_type": "pdf", "document": "s.pdf", "page": 7}}
    asked = {}

    def fake_retrieve(q, course, lecture=None, **kw):
        asked.update(q=q, k=kw["top_context"])
        return {"chunks": [hit] if q == "sampling" else []}
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    reply = json.dumps({"questions": [{"format": "short", "question": "Why sample?", "correct_answer": "x",
                                       "source_index": 1}]})
    prompts = []
    monkeypatch.setattr(qg, "chat_with_fallback", lambda m, **k: prompts.append(m[1]["content"]) or (reply, "groq"))
    out = qg.generate_questions("C", 1, counts={"Short Answer": 1}, topic=" sampling ", store=object())
    assert asked == {"q": "sampling", "k": 8} and out[0]["source"] == "s.pdf, p. 7"
    assert "Focus topic: sampling" in prompts[0]
    with pytest.raises(ValueError, match="Nothing about"):
        qg.generate_questions("C", 1, counts={"Short Answer": 1}, topic="quantum", store=object())


# --------------------------------------------------------------------------- tasks/evaluator
def test_concept_coverage_is_fuzzy_and_deterministic():
    from backend.tasks.evaluator import concept_coverage
    answer = "Örnekleme frekansı en yüksek frekansın iki katı olmalı, yoksa aliasing oluşur."
    assert concept_coverage(answer, ["aliasing", "örnekleme frekansı", "Nyquist oranı"]) == pytest.approx(2 / 3)
    assert concept_coverage(answer, []) is None


def test_concept_coverage_accepts_grader_judgement_for_paraphrases():
    from backend.tasks.evaluator import concept_coverage
    answer = "The values at the same index n are added sample by sample."
    concepts = [{"concept": "element-wise addition", "covered": True},   # paraphrase, judged covered
                {"concept": "same index", "covered": False},             # words present -> still counts
                {"concept": "commutativity", "covered": False}]
    assert concept_coverage(answer, concepts) == pytest.approx(2 / 3)


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


def test_reference_answer_reaches_the_grader():
    from backend.llm.prompts import build_eval_messages
    chunk = {"text": "u(n)=1 for n>=0", "metadata": {"source_type": "pdf", "document": "k.pdf", "page": 1}}
    with_ref = build_eval_messages("q", "a", [chunk], "u(n)=1 for n>=0, 0 for n<0")[1]["content"]
    assert "MODEL ANSWER" in with_ref and "0 for n<0" in with_ref
    assert "MODEL ANSWER" not in build_eval_messages("q", "a", [chunk])[1]["content"]


def test_evaluation_without_material_is_refused(monkeypatch):
    from backend.tasks import evaluator as ev
    monkeypatch.setattr(ev, "retrieve", lambda *a, **k: {"chunks": []})
    with pytest.raises(ValueError):
        ev.evaluate_answer("q", "a", "")
