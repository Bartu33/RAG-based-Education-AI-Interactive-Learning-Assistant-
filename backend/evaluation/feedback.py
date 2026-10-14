"""User feedback for the usability and student-satisfaction evaluation dimensions.

Two sources, each stored as JSON lines under `feedback.path` (git-ignored: it holds student text):
- ratings.jsonl: a thumbs up/down (+ optional comment) on one Q&A answer or one rubric evaluation
- sus.jsonl: System Usability Scale questionnaires (Brooke, 1996), scored 0-100
"""
import json
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import get_config, resolve_path

KINDS = ("qa", "evaluation")
SUS_ITEMS = 10
_LOCK = threading.Lock()


def _dir() -> Path:
    cfg = get_config().get("feedback") or {}
    return resolve_path(cfg.get("path", "data/feedback"))


def _append(name: str, record: dict) -> None:
    folder = _dir()
    folder.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with _LOCK, open(folder / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(name: str) -> list[dict]:
    path = _dir() / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_rating(kind: str, up: bool, comment: str = "", context: Optional[dict] = None,
               rating_id: Optional[str] = None) -> None:
    """rating_id identifies the rated item: a later record with the same id (changed thumb, added
    comment) replaces the earlier one in summary()."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    _append("ratings.jsonl", {"id": rating_id, "kind": kind, "up": bool(up),
                              "comment": (comment or "").strip(), **(context or {})})


def _latest_ratings() -> list[dict]:
    latest, anonymous = {}, []
    for r in _read("ratings.jsonl"):
        if r.get("id"):
            latest[r["id"]] = r
        else:
            anonymous.append(r)
    return anonymous + list(latest.values())


def sus_score(responses: list[int]) -> float:
    """Standard SUS: odd items contribute (r - 1), even items (5 - r); the sum times 2.5 -> 0..100."""
    if len(responses) != SUS_ITEMS or any(r not in (1, 2, 3, 4, 5) for r in responses):
        raise ValueError("SUS needs 10 answers, each from 1 to 5")
    total = sum((r - 1) if i % 2 == 0 else (5 - r) for i, r in enumerate(responses))
    return total * 2.5


def sus_grade(score: float) -> str:
    """Adjective rating (Bangor, Kortum & Miller, 2009); 68 is the commonly cited average."""
    for limit, label in ((85.5, "Excellent"), (72.6, "Good"), (52.0, "OK"), (39.2, "Poor")):
        if score >= limit:
            return label
    return "Awful"


def add_sus(responses: list[int], comment: str = "") -> float:
    score = sus_score(responses)
    _append("sus.jsonl", {"responses": responses, "score": score, "comment": (comment or "").strip()})
    return score


def summary() -> dict:
    ratings = _latest_ratings()
    by_kind = {}
    for kind in KINDS:
        rows = [r for r in ratings if r.get("kind") == kind]
        up = sum(1 for r in rows if r.get("up"))
        by_kind[kind] = {"count": len(rows), "up": up, "down": len(rows) - up,
                         "positive_pct": round(100 * up / len(rows), 1) if rows else None}
    scores = [s["score"] for s in _read("sus.jsonl")]
    mean = round(statistics.mean(scores), 1) if scores else None
    return {"ratings": by_kind,
            "sus": {"count": len(scores), "mean": mean,
                    "stdev": round(statistics.stdev(scores), 1) if len(scores) > 1 else None,
                    "grade": sus_grade(mean) if mean is not None else None},
            "comments": [{"kind": r["kind"], "up": r["up"], "comment": r["comment"], "time": r["time"]}
                         for r in ratings if r.get("comment")][-20:]}
