"""Student accounts: register / log in with student number + password, bearer-token sessions.

Passwords are stored as salted PBKDF2-SHA256 hashes (standard library); tokens are random and
expire after `SESSION_DAYS`.
"""
import hashlib
import hmac
import re
import secrets
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.db import LOCK, connect

STUDENT_NO = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
MIN_PASSWORD = 6
SESSION_DAYS = 30
_ITERATIONS = 200_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    _, iterations, salt, digest = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations)).hex()
    return hmac.compare_digest(check, digest)


def _new_session(con, student_no: str) -> str:
    token = secrets.token_urlsafe(32)
    con.execute("INSERT INTO sessions (token, student_no, expires) VALUES (?, ?, ?)",
                [token, student_no, (_now() + timedelta(days=SESSION_DAYS)).isoformat()])
    return token


def register(student_no: str, password: str) -> str:
    """Create the account and log it in. Returns a session token."""
    student_no = (student_no or "").strip()
    if not STUDENT_NO.match(student_no):
        raise ValueError("Student number: 3-32 letters, digits, '.', '_' or '-'")
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
    with LOCK, closing(connect()) as con, con:
        if con.execute("SELECT 1 FROM users WHERE student_no = ?", [student_no]).fetchone():
            raise ValueError("This student number is already registered; log in instead")
        con.execute("INSERT INTO users (student_no, password_hash, created) VALUES (?, ?, ?)",
                    [student_no, hash_password(password), _now().isoformat(timespec="seconds")])
        return _new_session(con, student_no)


def login(student_no: str, password: str) -> str:
    student_no = (student_no or "").strip()
    with LOCK, closing(connect()) as con, con:
        row = con.execute("SELECT password_hash FROM users WHERE student_no = ?", [student_no]).fetchone()
        if not row or not verify_password(password or "", row["password_hash"]):
            raise PermissionError("Wrong student number or password")
        return _new_session(con, student_no)


def student_for(token: Optional[str]) -> Optional[str]:
    """The logged-in student number, or None for a missing / unknown / expired token."""
    if not token:
        return None
    with closing(connect()) as con:
        row = con.execute("SELECT student_no, expires FROM sessions WHERE token = ?", [token]).fetchone()
    if not row or datetime.fromisoformat(row["expires"]) < _now():
        return None
    return row["student_no"]


def logout(token: str) -> None:
    with LOCK, closing(connect()) as con, con:
        con.execute("DELETE FROM sessions WHERE token = ?", [token])
