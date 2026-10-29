import random
import uuid
from typing import Optional

from backend.indexing.vectorstore import VectorStore, and_filter, get_store
from backend.llm.prompts import build_question_messages, exam_style_block
from backend.llm.providers import chat_with_fallback, parse_json, resolve_provider_name
from backend.schemas import format_location

MAX_QUESTIONS = 20
MAX_PER_TYPE = 10
# context grows with the test so that many questions do not all come from the same few excerpts
MIN_CONTEXT_CHUNKS, MAX_CONTEXT_CHUNKS = 8, 16
FORMAT_ORDER = ("mcq", "tf", "short", "concept")
CONCEPT_EXTRA_TOKENS = 100  # a worked calculation is longer than a conceptual answer


def _canonical(label: str) -> str:
    low = label.lower()
    if "multiple" in low or "choice" in low or "mcq" in low:
        return "mcq"
    if "true" in low or "false" in low or low == "tf":
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


def _concept_fields(q: dict, fmt: str) -> tuple[Optional[str], Optional[str]]:
    """(kind, final_answer) of a concept question; (None, None) for the other formats.
    kind is 'calculation' or 'conceptual'; final_answer is the result of a calculation."""
    if fmt != "concept":
        return None, None
    kind = "calculation" if str(q.get("kind") or "").strip().lower().startswith("calc") else "conceptual"
    final = str(q.get("final_answer") or "").strip() or None
    return kind, final if kind == "calculation" else None


def make_plan(counts: Optional[dict] = None, types: Optional[list[str]] = None,
              n: Optional[int] = None) -> list[tuple[str, str]]:
    """[(format, UI label), ...]. `counts` ({"Multiple Choice": 5, "True/False": 1, ...}) wins;
    otherwise the older `types` + `n` request cycles through the types."""
    if counts:
        per = {}
        for label, k in counts.items():
            k = int(k or 0)
            if k < 0 or k > MAX_PER_TYPE:
                raise ValueError(f"Between 0 and {MAX_PER_TYPE} questions per type")
            if k:
                per[_canonical(label)] = (label, per.get(_canonical(label), ("", 0))[1] + k)
        plan = [(fmt, per[fmt][0]) for fmt in FORMAT_ORDER if fmt in per for _ in range(per[fmt][1])]
    else:
        if not types or not n:
            raise ValueError("Choose at least one question type")
        plan = [(_canonical(types[i % len(types)]), types[i % len(types)]) for i in range(int(n))]
    if not plan:
        raise ValueError("Choose at least one question")
    if len(plan) > MAX_QUESTIONS:
        raise ValueError(f"At most {MAX_QUESTIONS} questions per test")
    return plan


def _search(query: str, course: str, lecture: Optional[int], k: int, store: VectorStore) -> list[dict]:
    from backend.retrieval.retriever import retrieve
    return retrieve(query, course, lecture, top_candidates=max(20, 2 * k), top_context=k, store=store)["chunks"]


def _context(course: str, lecture: Optional[int], topic: Optional[str], k: int,
             store: VectorStore, exam_topics: Optional[list[str]] = None) -> list[dict]:
    """With a topic: the excerpts the Q&A search finds for it. With past exams: excerpts about the
    topics the exams ask about. Otherwise (or when those find nothing): a random sample."""
    if topic:
        chunks = _search(topic, course, lecture, k, store)
        if not chunks:
            raise ValueError(f"Nothing about '{topic}' is indexed for this course/lecture.")
        return chunks
    if exam_topics:
        per = -(-k // len(exam_topics))
        found = {}
        for t in exam_topics:
            for c in _search(t, course, lecture, per, store):
                found.setdefault(c["text"], c)
        if found:
            return list(found.values())[:k]
    where = and_filter([{"course": {"$eq": course}},
                        {"lecture": {"$eq": int(lecture)}} if lecture else None])
    pool = store.get(where=where, limit=500)
    if not pool:
        raise ValueError("No indexed material for this course/lecture.")
    return random.sample(pool, min(k, len(pool)))


def generate_questions(course: str, lecture: Optional[int], types: Optional[list[str]] = None,
                       n: Optional[int] = None, difficulty: str = "Medium",
                       provider: Optional[str] = None, store: Optional[VectorStore] = None,
                       counts: Optional[dict] = None, topic: Optional[str] = None,
                       exam_ids: Optional[list[str]] = None) -> list[dict]:
    """exam_ids: past exams of the course (backend/tasks/exam_style.py) whose style to imitate."""
    plan = make_plan(counts, types, n)
    topic = (topic or "").strip() or None
    k = min(MAX_CONTEXT_CHUNKS, max(MIN_CONTEXT_CHUNKS, len(plan)))
    style, examples, topics = None, [], None
    if exam_ids:
        from backend.tasks import exam_style
        exams = exam_style.get_exams(exam_ids, course)
        examples = exam_style.pick_examples(exams)
        style = exam_style_block(exams, examples)
        topics = exam_style.exam_topics(exams)
    chunks = _context(course, lecture, topic, k, store or get_store(), topics)

    text, _ = chat_with_fallback(build_question_messages(chunks, [f for f, _ in plan], difficulty, topic, style),
                                 provider=resolve_provider_name(provider), temperature=0.5,
                                 max_tokens=500 + 250 * len(plan)
                                 + CONCEPT_EXTRA_TOKENS * sum(1 for f, _ in plan if f == "concept"),
                                 json_mode=True)
    raw = parse_json(text).get("questions", [])

    # check each item against the format it says it has (falls back to its position), so a model
    # that reorders the questions does not lose valid ones; keep at most the requested number per type
    wanted = {fmt: sum(1 for f, _ in plan if f == fmt) for fmt, _ in plan}
    label = {fmt: lab for fmt, lab in plan}
    out = []
    for i, q in enumerate(raw):
        stated = q.get("format")
        fmt = _canonical(str(stated)) if stated else (plan[i][0] if i < len(plan) else None)
        if fmt not in wanted or wanted[fmt] == 0 or not _valid(q, fmt):
            continue
        wanted[fmt] -= 1
        src = q.get("source_index")
        meta = chunks[src - 1]["metadata"] if isinstance(src, int) and 1 <= src <= len(chunks) \
            else chunks[0]["metadata"]
        kind, final_answer = _concept_fields(q, fmt)
        out.append({"id": uuid.uuid4().hex[:8], "type": label[fmt], "difficulty": difficulty,
                    "question": q["question"], "options": q.get("options") if fmt == "mcq" else None,
                    "correct_answer": q["correct_answer"],
                    "explanation": q.get("explanation", ""), "source": format_location(meta),
                    "kind": kind, "final_answer": final_answer})
        ref = q.get("similar_to")
        if examples and isinstance(ref, int) and 1 <= ref <= len(examples):
            ex = examples[ref - 1]
            out[-1]["style_ref"] = f"{ex['exam']}: {ex['text'][:120]}{'…' if len(ex['text']) > 120 else ''}"
    out.sort(key=lambda q: FORMAT_ORDER.index(_canonical(q["type"])))  # grouped like the request
    return out
