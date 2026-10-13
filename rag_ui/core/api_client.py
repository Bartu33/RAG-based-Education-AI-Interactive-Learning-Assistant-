"""HTTP client for the FastAPI backend (see docs/api_contract.md).

The UI never imports `backend.*`: ChromaDB must not be opened by two processes, so every call goes
through the API. This module has no Streamlit dependency so it can be tested on its own.
"""
import os
from typing import Optional

import httpx

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)    # LLM calls (ask / generate / evaluate)
LONG = httpx.Timeout(None, connect=5.0)         # ingest of a long video, benchmark
_client: Optional[httpx.Client] = None


class ApiError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def use_client(client: Optional[httpx.Client]) -> None:
    """Swap the HTTP client (tests pass a FastAPI TestClient)."""
    global _client
    _client = client


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
    return _client


def _call(method: str, path: str, **kwargs):
    try:
        r = _http().request(method, path, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # Windows retries a refused connection for ~2 s, so "offline" can also arrive as a timeout
        raise ApiError(f"Cannot reach the backend at {BASE_URL}. "
                       "Start it with: uvicorn backend.main:app")
    except httpx.TimeoutException:
        raise ApiError("The backend did not answer in time.")
    except httpx.HTTPError:
        raise ApiError(f"Cannot reach the backend at {BASE_URL}. "
                       "Start it with: uvicorn backend.main:app")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # 422 validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(str(detail or r.reason_phrase), r.status_code)
    return r.json()


# --- status / settings ---
def health() -> dict:
    return _call("GET", "/health")


def public_config() -> dict:
    return _call("GET", "/config")


def stats() -> dict:
    return _call("GET", "/stats")


# --- materials ---
def courses() -> list[dict]:
    return _call("GET", "/courses")["courses"]


def documents() -> list[dict]:
    return _call("GET", "/documents")["documents"]


def ingest_file(name: str, data: bytes, course: str, lecture: int,
                mime: str = "application/octet-stream") -> dict:
    res = _call("POST", "/ingest", data={"course": course, "lecture": str(lecture)},
                files=[("files", (name, data, mime))], timeout=LONG)
    return res["records"][0]


def delete_course(course: str, lecture: Optional[int] = None) -> int:
    params = {"lecture": lecture} if lecture else None
    return _call("DELETE", f"/courses/{course}", params=params)["deleted"]


def clear_index() -> dict:
    return _call("DELETE", "/index")


def low_confidence(threshold: Optional[float] = None, course: Optional[str] = None,
                   limit: int = 50) -> list[dict]:
    params = {"limit": limit}
    if threshold is not None:
        params["threshold"] = threshold
    if course:
        params["course"] = course
    return _call("GET", "/transcripts/low-confidence", params=params)["segments"]


def correct_chunk(chunk_id: str, text: str) -> dict:
    return _call("PUT", f"/chunks/{chunk_id}", json={"text": text})


# --- tasks ---
def ask(question: str, course: str, lecture: Optional[int] = None,
        ts_range: Optional[tuple] = None, cfg: Optional[dict] = None) -> dict:
    cfg = cfg or {}
    body = {"question": question, "course": course, "lecture": lecture,
            "ts_range": list(ts_range) if ts_range else None,
            "config": {k: cfg[k] for k in ("provider", "top_k_candidates", "top_k_context")
                       if k in cfg}}
    return _call("POST", "/ask", json=body)


def generate_questions(course: str, lecture: Optional[int], types: list[str], n: int,
                       difficulty: str, provider: Optional[str] = None) -> list[dict]:
    body = {"course": course, "lecture": lecture, "types": types, "n": n,
            "difficulty": difficulty, "provider": provider}
    return _call("POST", "/generate", json=body)["questions"]


def evaluate(question: str, answer: str, course: str, lecture: Optional[int] = None,
             provider: Optional[str] = None) -> dict:
    body = {"question": question, "answer": answer, "course": course, "lecture": lecture,
            "provider": provider}
    return _call("POST", "/evaluate", json=body)


def benchmark(sizes: Optional[list[int]] = None, provider: Optional[str] = None) -> list[dict]:
    return _call("POST", "/benchmark", json={"sizes": sizes, "provider": provider},
                 timeout=LONG)["rows"]
