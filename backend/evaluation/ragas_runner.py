"""RAGAs-style, reference-free metrics implemented natively (no extra dependency).
Faithfulness follows the RAGAs idea: split the answer into claims, check each against the context.
Thresholds come from config.yaml and are to be calibrated with expert labels in WP4."""
from typing import Optional

import numpy as np

from backend.config import get_config
from backend.indexing.embedder import embed_texts
from backend.llm.prompts import build_faithfulness_messages, build_hallucination_messages
from backend.llm.providers import chat_with_fallback, parse_json


def faithfulness_score(answer: str, contexts: list[str], provider: Optional[str] = None) -> float:
    """Share of answer claims supported by the context (0..1). No claims -> 1.0."""
    text, _ = chat_with_fallback(build_faithfulness_messages(answer, contexts),
                                 provider=provider, temperature=0.0,
                                 max_tokens=800, json_mode=True)
    claims = parse_json(text).get("claims", [])
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if c.get("supported") is True)
    return round(supported / len(claims), 3)


def answer_relevancy(question: str, answer: str) -> float:
    """Embedding cosine between question and answer (cheap proxy for RAGAs answer relevancy)."""
    emb = embed_texts([question, answer])
    return round(float(np.clip(np.dot(emb[0], emb[1]), 0.0, 1.0)), 3)


def classify(faithfulness: float) -> str:
    cfg = get_config()["grounding"]
    if faithfulness >= cfg["grounded_threshold"]:
        return "grounded"
    if faithfulness >= cfg["partial_threshold"]:
        return "partially_grounded"
    return "unsupported"


def hallucination_type(answer: str, contexts: list[str], provider: Optional[str] = None) -> str:
    """intrinsic (contradicts source) | extrinsic (unverifiable) | none  (Ji et al., 2023)."""
    text, _ = chat_with_fallback(build_hallucination_messages(answer, contexts),
                                 provider=provider, temperature=0.0,
                                 max_tokens=60, json_mode=True)
    value = str(parse_json(text).get("type", "none")).lower()
    return value if value in ("intrinsic", "extrinsic", "none") else "none"


def evaluate_sample(question: str, answer: str, contexts: list[str],
                    provider: Optional[str] = None) -> dict:
    faith = faithfulness_score(answer, contexts, provider)
    label = classify(faith)
    return {"faithfulness": faith,
            "answer_relevancy": answer_relevancy(question, answer),
            "label": label,
            "hallucination_type": None if label == "grounded"
            else hallucination_type(answer, contexts, provider)}