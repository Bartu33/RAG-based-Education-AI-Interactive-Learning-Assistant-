"""SQLite store for application records: student accounts, login sessions, practice tests and answers.

Python standard library, one file at `database.path` (git-ignored: it holds password hashes and
student answers). ChromaDB stays for what it is built for, vector search over course material.
"""
import sqlite3
import threading

from backend.config import get_config, resolve_path

LOCK = threading.Lock()   # serialises writes from FastAPI's worker threads
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    student_no    TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,          -- pbkdf2_sha256$iterations$salt$hash
    created       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    student_no TEXT NOT NULL REFERENCES users(student_no) ON DELETE CASCADE,
    expires    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tests (
    id         TEXT PRIMARY KEY,
    student_no TEXT NOT NULL REFERENCES users(student_no) ON DELETE CASCADE,
    created    TEXT NOT NULL,
    course     TEXT NOT NULL,
    lecture    INTEGER,
    difficulty TEXT,
    questions  TEXT NOT NULL              -- JSON list of the generated questions
);
CREATE TABLE IF NOT EXISTS attempts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id        TEXT NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    question_id    TEXT NOT NULL,
    time           TEXT NOT NULL,
    student_answer TEXT NOT NULL,
    is_correct     INTEGER,               -- choice questions: 1 / 0
    score          INTEGER,               -- open questions: rubric total 0-100
    feedback       TEXT,                  -- JSON list of feedback sentences
    dims           TEXT                   -- JSON list of rubric dimensions
);
CREATE INDEX IF NOT EXISTS tests_student ON tests (student_no, created);
CREATE INDEX IF NOT EXISTS attempts_test ON attempts (test_id, question_id);
"""


def connect() -> sqlite3.Connection:
    path = resolve_path((get_config().get("database") or {}).get("path", "data/app.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con
