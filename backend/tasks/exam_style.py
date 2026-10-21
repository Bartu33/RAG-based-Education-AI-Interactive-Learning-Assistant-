"""Past exams of a course: the instructor's way of asking questions.

An uploaded exam is read once by the LLM, which lists its questions and describes their style
(phrasing, structure, calculation vs. theory, typical topics). Question generation can then write
new questions in that style, from the course material and about the topics the exams focus on.

Exams are kept in SQLite (backend/db.py), not in the vector index: they are style examples, not
course material, so Q&A and the study buddy never cite them.
"""
import json
import random
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.db import LOCK, connect
from backend.llm.prompts import build_exam_style_messages
from backend.llm.providers import chat_with_fallback, parse_json, resolve_provider_name

MAX_EXAM_CHARS = 12_000      # ~5-6 exam pages; keeps one analysis within free-tier rate limits
MAX_EXAM_QUESTIONS = 30
MAX_QUESTION_CHARS = 600
FORMATS = ("mcq", "tf", "short", "concept")
TEXT_TYPES = (".pdf", ".docx", ".pptx")


def extract_text(path) -> str:
    """Text of an exam file (pdf / docx / pptx)."""
    from backend.ingestion import load_file
    if Path(path).suffix.lower() not in TEXT_TYPES:
        raise ValueError("Past exams can be PDF, DOCX or PPTX files")
    text = "\n\n".join(s.text for s in load_file(path)).strip()
    if not text:
        raise ValueError("No text found in this file (a scanned exam?). Paste its questions as text instead.")
    return text


def _fmt(value) -> str:
    low = str(value or "").lower()
    return low if low in FORMATS else "concept"


def normalize_style(raw: dict) -> dict:
    """Keep only well-formed parts of the model's analysis."""
    questions = []
    for q in raw.get("questions") or []:
        text = str((q or {}).get("text", "")).strip() if isinstance(q, dict) else ""
        if text:
            questions.append({"text": text[:MAX_QUESTION_CHARS], "topic": str(q.get("topic") or "").strip(),
                              "format": _fmt(q.get("format"))})
    questions = questions[:MAX_EXAM_QUESTIONS]
    mix = {f: sum(1 for q in questions if q["format"] == f) for f in FORMATS}
    return {"summary": str(raw.get("summary") or "").strip(),
            "style_points": [str(p).strip() for p in raw.get("style_points") or [] if str(p).strip()][:8],
            "topics": [str(t).strip() for t in raw.get("topics") or [] if str(t).strip()][:12],
            "mix": mix, "questions": questions}


def analyze(text: str, provider: Optional[str] = None) -> dict:
    reply, _ = chat_with_fallback(build_exam_style_messages(text[:MAX_EXAM_CHARS]),
                                  provider=resolve_provider_name(provider), temperature=0.2,
                                  max_tokens=3000, json_mode=True)
    style = normalize_style(parse_json(reply))
    if not style["questions"]:
        raise ValueError("No exam questions were recognised in this text.")
    return style


def _public(row) -> dict:
    style = json.loads(row["style"])
    return {"id": row["id"], "course": row["course"], "name": row["name"], "created": row["created"],
            "truncated": len(row["text"]) > MAX_EXAM_CHARS, **style}


def save_exam(course: str, name: str, text: str, provider: Optional[str] = None) -> dict:
    course, name, text = (course or "").strip(), (name or "").strip(), (text or "").strip()
    if not course or not text:
        raise ValueError("An exam needs a course and its text")
    style = analyze(text, provider)
    exam_id = uuid.uuid4().hex[:12]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOCK, closing(connect()) as con, con:
        con.execute("INSERT INTO exams (id, course, name, created, text, style) VALUES (?, ?, ?, ?, ?, ?)",
                    [exam_id, course, name or f"Exam {created[:10]}", created, text,
                     json.dumps(style, ensure_ascii=False)])
        row = con.execute("SELECT * FROM exams WHERE id = ?", [exam_id]).fetchone()
    return _public(row)


def list_exams(course: str) -> list[dict]:
    with closing(connect()) as con:
        rows = con.execute("SELECT * FROM exams WHERE course = ? ORDER BY created", [course]).fetchall()
    return [_public(r) for r in rows]


def get_exams(ids: list[str], course: str) -> list[dict]:
    exams = [e for e in list_exams(course) if e["id"] in set(ids)]
    if len(exams) != len(set(ids)):
        raise ValueError("Some of the chosen past exams do not belong to this course")
    return exams


def delete_exam(exam_id: str) -> bool:
    with LOCK, closing(connect()) as con, con:
        return con.execute("DELETE FROM exams WHERE id = ?", [exam_id]).rowcount > 0


def pick_examples(exams: list[dict], k: int = 8, rng=random) -> list[dict]:
    """Up to k past questions, taken in turn from each exam: [{"exam", "text", "topic", "format"}]."""
    pools = [[{**q, "exam": e["name"]} for q in e["questions"]] for e in exams]
    for p in pools:
        rng.shuffle(p)
    out = []
    while len(out) < k and any(pools):
        for p in pools:
            if p and len(out) < k:
                out.append(p.pop())
    return out


def exam_topics(exams: list[dict], n: int = 4, rng=random) -> list[str]:
    """Topics the exams ask about, the most frequent ones more likely to be picked."""
    seen = [q["topic"] for e in exams for q in e["questions"] if q["topic"]]
    seen += [t for e in exams for t in e["topics"]]
    rng.shuffle(seen)
    return list(dict.fromkeys(seen))[:n]
