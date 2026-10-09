import re
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

from backend.config import get_config
from backend.indexing.embedder import embed_texts
from backend.llm.prompts import build_eval_messages
from backend.llm.providers import chat_with_fallback, parse_json, resolve_provider_name
from backend.retrieval.retriever import retrieve
from backend.schemas import format_location

_WORD = re.compile(r"\w+", re.UNICODE)


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _covered(concept: str, answer_tokens: list[str]) -> bool:
    parts = _tokens(concept)
    return bool(parts) and all(
        any(SequenceMatcher(None, p, a).ratio() >= 0.85 for a in answer_tokens) for p in parts)


def concept_coverage(answer: str, concepts: list[str]) -> Optional[float]:
    """Deterministic: share of key concepts mentioned in the answer (fuzzy token match)."""
    if not concepts:
        return None
    toks = _tokens(answer)
    return sum(_covered(c, toks) for c in concepts) / len(concepts)


def terminology_score(answer: str, context: str) -> float:
    """Deterministic: mean of TF-IDF cosine and embedding cosine between answer and course context.
    Raw similarity scale, so calibrate against expert labels (WP4)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        m = TfidfVectorizer().fit_transform([context, answer])
        tfidf = float(m[0].multiply(m[1]).sum())
    except ValueError:  # empty vocabulary
        tfidf = 0.0
    emb = embed_texts([answer, context])
    return _clip(0.5 * tfidf + 0.5 * float(np.dot(emb[0], emb[1])))


def evaluate_answer(question: str, student_answer: str, course: str,
                    lecture: Optional[int] = None, provider: Optional[str] = None) -> dict:
    r = retrieve(question, course, lecture)
    chunks = r["chunks"]
    if not chunks:
        raise ValueError("No indexed material found for this question.")

    text, _ = chat_with_fallback(build_eval_messages(question, student_answer, chunks),
                                 provider=resolve_provider_name(provider),
                                 temperature=0.0, max_tokens=900, json_mode=True)
    data = parse_json(text)
    accuracy = _clip(float(data.get("factual_accuracy", 0)))
    completeness = _clip(float(data.get("completeness", 0)))
    coverage = concept_coverage(student_answer, data.get("key_concepts", []))
    coverage = completeness if coverage is None else coverage
    terminology = terminology_score(student_answer, "\n".join(c["text"] for c in chunks))

    w = get_config()["rubric"]
    dims = [
        {"name": "Factual accuracy", "weight": w["factual_accuracy"], "score": round(accuracy, 2)},
        {"name": "Key concept coverage", "weight": w["concept_coverage"], "score": round(coverage, 2)},
        {"name": "Course terminology", "weight": w["terminology"], "score": round(terminology, 2)},
        {"name": "Completeness", "weight": w["completeness"], "score": round(completeness, 2)},
    ]
    feedback = data.get("feedback", [])
    return {"total": round(sum(d["weight"] * d["score"] for d in dims) * 100),
            "dims": dims,
            "feedback": feedback if isinstance(feedback, list) else [str(feedback)],
            "source": format_location(chunks[0]["metadata"]),
            "follow_up_question": data.get("follow_up_question", "")}