"""Practice history: every generated test and the answers given to it, per student (backend/db.py).

A test is one "Generate questions" run; its questions are stored with the test, so My Results can
show unanswered questions too. When a question is answered more than once, the latest answer counts.
"""
import json
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

from backend.db import LOCK, connect

_QUESTION_KEYS = ("id", "type", "difficulty", "question", "options", "correct_answer", "explanation", "source",
                  "style_ref", "kind", "final_answer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_test(student_no: str, course: str, lecture: Optional[int], difficulty: Optional[str],
                questions: list[dict]) -> str:
    if not course or not questions:
        raise ValueError("A test needs a course and at least one question")
    kept = [{k: q.get(k) for k in _QUESTION_KEYS} for q in questions]
    if any(not q["id"] or not q["question"] for q in kept):
        raise ValueError("Every question needs an id and a text")
    test_id = uuid.uuid4().hex
    with LOCK, closing(connect()) as con, con:
        con.execute("INSERT INTO tests (id, student_no, created, course, lecture, difficulty, questions) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [test_id, student_no, _now(), course, lecture, difficulty, json.dumps(kept, ensure_ascii=False)])
    return test_id


def add_attempt(student_no: str, test_id: str, question_id: str, student_answer: str,
                is_correct: Optional[bool] = None, score: Optional[int] = None,
                feedback: Optional[list] = None, dims: Optional[list] = None) -> int:
    if not (student_answer or "").strip():
        raise ValueError("student_answer is required")
    if is_correct is None and score is None:
        raise ValueError("an attempt needs is_correct (choice) or score (open question)")
    with LOCK, closing(connect()) as con, con:
        row = con.execute("SELECT questions FROM tests WHERE id = ? AND student_no = ?",
                          [test_id, student_no]).fetchone()
        if not row:
            raise LookupError("No such test for this student")
        if question_id not in {q["id"] for q in json.loads(row["questions"])}:
            raise ValueError("The question is not part of this test")
        cur = con.execute(
            "INSERT INTO attempts (test_id, question_id, time, student_answer, is_correct, score, feedback, dims) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [test_id, question_id, _now(), student_answer, None if is_correct is None else int(bool(is_correct)),
             score, json.dumps(feedback, ensure_ascii=False) if feedback is not None else None,
             json.dumps(dims, ensure_ascii=False) if dims is not None else None])
        return cur.lastrowid


def _summary(questions: list[dict]) -> dict:
    answered = [q for q in questions if q["attempt"]]
    choice = [q["attempt"]["is_correct"] for q in answered if q["attempt"]["is_correct"] is not None]
    scores = [q["attempt"]["score"] for q in answered if q["attempt"]["score"] is not None]
    return {"questions": len(questions), "answered": len(answered),
            "choice": {"count": len(choice), "correct": sum(choice)},
            "open": {"count": len(scores), "average": round(sum(scores) / len(scores), 1) if scores else None}}


def list_tests(student_no: str) -> dict:
    """{"tests": newest first, each with its questions and the latest attempt per question, "summary"}."""
    with closing(connect()) as con:
        tests = con.execute("SELECT * FROM tests WHERE student_no = ? ORDER BY created, rowid",
                            [student_no]).fetchall()
        attempts = con.execute(
            "SELECT a.* FROM attempts a JOIN tests t ON t.id = a.test_id WHERE t.student_no = ? ORDER BY a.id",
            [student_no]).fetchall()
    latest, tries = {}, {}
    for a in attempts:  # ordered by id, so the last one per question wins
        key = (a["test_id"], a["question_id"])
        tries[key] = tries.get(key, 0) + 1
        latest[key] = {"time": a["time"], "student_answer": a["student_answer"],
                       "is_correct": None if a["is_correct"] is None else bool(a["is_correct"]),
                       "score": a["score"], "feedback": json.loads(a["feedback"]) if a["feedback"] else None,
                       "dims": json.loads(a["dims"]) if a["dims"] else None}
    out = []
    for number, t in enumerate(tests, 1):
        questions = json.loads(t["questions"])
        for q in questions:
            key = (t["id"], q["id"])
            q["attempt"] = {**latest[key], "tries": tries[key]} if key in latest else None
        out.append({"id": t["id"], "number": number, "created": t["created"], "course": t["course"],
                    "lecture": t["lecture"], "difficulty": t["difficulty"], "questions": questions,
                    "summary": _summary(questions)})
    out.reverse()
    every = [q for t in out for q in t["questions"]]
    return {"tests": out, "summary": {"tests": len(out), **_summary(every)}}


def delete_history(student_no: str) -> int:
    with LOCK, closing(connect()) as con, con:
        return con.execute("DELETE FROM tests WHERE student_no = ?", [student_no]).rowcount
