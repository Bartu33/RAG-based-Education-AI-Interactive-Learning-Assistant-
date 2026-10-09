import random
import uuid
from typing import Optional

from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.llm.prompts import build_question_messages
from backend.llm.providers import chat_with_fallback, parse_json, resolve_provider_name
from backend.schemas import format_location

CONTEXT_CHUNKS = 8


def _canonical(label: str) -> str:
    low = label.lower()
    if "multiple" in low or "choice" in low or "mcq" in low:
        return "mcq"
    if "true" in low or "false" in low:
        return "tf"
    if "short" in low:
        return "short"
    return "concept"


def _valid(q: dict, fmt: str) -> bool:
    if not q.get("question") or not q.get("correct_answer"):
        return False
    if fmt == "mcq":
        opts = q.get("options")
        return isinstance(opts, list) and len(opts) == 4 and q["correct_answer"] in opts
    if fmt == "tf":
        return str(q["correct_answer"]).strip().lower() in ("true", "false")
    return True


def generate_questions(course: str, lecture: Optional[int], types: list[str], n: int,
                       difficulty: str, provider: Optional[str] = None,
                       store: Optional[VectorStore] = None) -> list[dict]:
    store = store or get_store()
    where = and_filter([{"course": {"$eq": course}},
                        {"lecture": {"$eq": int(lecture)}} if lecture else None])
    pool = store.get(where=where, limit=500)
    if not pool:
        raise ValueError("No indexed material for this course/lecture.")
    chunks = random.sample(pool, min(CONTEXT_CHUNKS, len(pool)))

    plan_labels = [types[i % len(types)] for i in range(n)]
    plan = [_canonical(t) for t in plan_labels]
    text, _ = chat_with_fallback(build_question_messages(chunks, plan, difficulty),
                                 provider=resolve_provider_name(provider),
                                 temperature=0.5, max_tokens=2500, json_mode=True)
    raw = parse_json(text).get("questions", [])

    out = []
    for i, q in enumerate(raw[:n]):
        if not _valid(q, plan[i]):
            continue
        src = q.get("source_index")
        meta = chunks[src - 1]["metadata"] if isinstance(src, int) and 1 <= src <= len(chunks) \
            else chunks[0]["metadata"]
        out.append({"id": uuid.uuid4().hex[:8], "type": plan_labels[i], "difficulty": difficulty,
                    "question": q["question"], "options": q.get("options"),
                    "correct_answer": q["correct_answer"],
                    "explanation": q.get("explanation", ""), "source": format_location(meta)})
    return out