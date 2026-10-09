from time import perf_counter
from typing import Optional

from backend.config import get_config
from backend.evaluation.ragas_runner import faithfulness_score
from backend.indexing.chunker import count_tokens
from backend.indexing.vectorstore import VectorStore
from backend.llm.prompts import build_qa_messages
from backend.llm.providers import chat_with_fallback, resolve_provider_name
from backend.retrieval.retriever import retrieve
from backend.schemas import chunk_to_source

NO_INFO = "I could not find this in the approved course materials."


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
        return {"answer": NO_INFO, "faithfulness": 1.0, "sources": [], "contexts": [],
                "retrieval_ms": r["retrieval_ms"], "total_s": round(perf_counter() - t0, 2),
                "tokens_sent": 0, "candidates": r["candidates"], "filtered_from": r["filtered_from"]}

    messages = build_qa_messages(question, chunks)
    answer, used = chat_with_fallback(messages, provider=provider)
    contexts = [c["text"] for c in chunks]

    faith = None  # if disabled in config.yaml the UI must handle None
    if get_config()["qa"]["score_faithfulness"]:
        faith = faithfulness_score(answer, contexts, used)

    return {"answer": answer, "faithfulness": faith,
            "sources": [chunk_to_source(c) for c in chunks], "contexts": contexts,
            "retrieval_ms": r["retrieval_ms"], "total_s": round(perf_counter() - t0, 2),
            "tokens_sent": sum(count_tokens(m["content"]) for m in messages),
            "candidates": r["candidates"], "filtered_from": r["filtered_from"],
            "provider": used}