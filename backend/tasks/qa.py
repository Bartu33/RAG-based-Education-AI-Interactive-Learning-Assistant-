from time import perf_counter
from typing import Optional

from backend.config import get_config
from backend.evaluation.ragas_runner import faithfulness_score
from backend.indexing.chunker import count_tokens
from backend.indexing.vectorstore import VectorStore
from backend.llm.prompts import build_qa_messages, split_beyond
from backend.llm.providers import chat_with_fallback, resolve_provider_name
from backend.retrieval.retriever import retrieve
from backend.schemas import chunk_to_source

NO_INFO = "I could not find this in the approved course materials."
NO_INFO_MARKER = "[NO_INFO]"   # the QA prompt asks the model to start "not covered" replies with it


def split_no_info(answer: str) -> tuple[str, bool]:
    """(answer without the marker, found). found=False when the model said the material lacks it."""
    if NO_INFO_MARKER not in answer:
        return answer, True
    return answer.replace(NO_INFO_MARKER, "").strip() or NO_INFO, False


def answer_question(question: str, course: str, lecture: Optional[int] = None,
                    ts_range_ms: Optional[tuple[int, int]] = None, cfg: Optional[dict] = None,
                    store: Optional[VectorStore] = None) -> dict:
    """Grounded Q&A. ts_range_ms = (start, end) of the video segment being watched."""
    overrides = cfg or {}
    provider = resolve_provider_name(overrides.get("provider"))
    t0 = perf_counter()

    r = retrieve(question, course, lecture, ts_range_ms,
                 top_candidates=overrides.get("top_k_candidates"),
                 top_context=overrides.get("top_k_context"), store=store)
    chunks = r["chunks"]
    if not chunks:  # nothing retrieved: nothing claimed, so nothing hallucinated
        return {"answer": NO_INFO, "found": False, "beyond": None, "faithfulness": 1.0, "sources": [], "contexts": [],
                "retrieval_ms": r["retrieval_ms"], "total_s": round(perf_counter() - t0, 2),
                "tokens_sent": 0, "candidates": r["candidates"], "filtered_from": r["filtered_from"]}

    beyond_on = get_config()["qa"].get("general_knowledge", True)
    messages = build_qa_messages(question, chunks, beyond=beyond_on)
    answer, used = chat_with_fallback(messages, provider=provider)
    # the [BEYOND] part (general knowledge / reasoning) is shown apart and never scored as grounded
    answer, beyond = split_beyond(answer) if beyond_on else (answer, None)
    answer, found = split_no_info(answer)
    contexts = [c["text"] for c in chunks]

    faith = None  # if disabled in config.yaml the UI must handle None
    if not found:
        faith = 1.0  # a "not covered" reply makes no claim, so there is nothing to verify
    elif overrides.get("score_faithfulness", get_config()["qa"]["score_faithfulness"]):
        faith = faithfulness_score(answer, contexts, used)

    return {"answer": answer, "found": found, "beyond": beyond, "faithfulness": faith,
            "sources": [chunk_to_source(c) for c in chunks], "contexts": contexts,
            "retrieval_ms": r["retrieval_ms"], "total_s": round(perf_counter() - t0, 2),
            "tokens_sent": sum(count_tokens(m["content"]) for m in messages),
            "candidates": r["candidates"], "filtered_from": r["filtered_from"],
            "provider": used}