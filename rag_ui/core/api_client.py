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


def generate_questions(course: str, lecture: Optional[int], counts: dict[str, int],
                       difficulty: str, provider: Optional[str] = None,
                       topic: Optional[str] = None, exam_ids: Optional[list[str]] = None) -> list[dict]:
    """counts: questions per type, e.g. {"Multiple Choice": 5, "True/False": 1}.
    exam_ids: past exams whose question style to imitate."""
    body = {"course": course, "lecture": lecture, "counts": counts, "topic": topic,
            "difficulty": difficulty, "provider": provider, "exam_ids": exam_ids or None}
    return _call("POST", "/generate", json=body)["questions"]


# --- past exams (instructor's question style) ---
def exams(course: str) -> list[dict]:
    return _call("GET", "/exams", params={"course": course})["exams"]


def add_exam(course: str, name: str = "", text: str = "", file: Optional[tuple] = None,
             provider: Optional[str] = None) -> dict:
    """file: (filename, bytes, mime). Either a file or pasted text."""
    data = {"course": course, "name": name, "text": text}
    if provider:
        data["provider"] = provider
    return _call("POST", "/exams", data=data, files=[("file", file)] if file else None)


def delete_exam(exam_id: str) -> dict:
    return _call("DELETE", f"/exams/{exam_id}")


def evaluate(question: str, answer: str, course: str, lecture: Optional[int] = None,
             provider: Optional[str] = None, reference_answer: Optional[str] = None) -> dict:
    body = {"question": question, "answer": answer, "course": course, "lecture": lecture,
            "provider": provider, "reference_answer": reference_answer}
    return _call("POST", "/evaluate", json=body)


# --- Interactive Learning (one open document + study-buddy chat) ---
def study_document(course: str, document: str) -> dict:
    return _call("GET", "/study/document", params={"course": course, "document": document})


def study_page_image(course: str, document: str, page: int, zoom: float = 1.5) -> bytes:
    """PNG of one PDF page."""
    params = {"course": course, "document": document, "page": page, "zoom": zoom}
    try:
        r = _http().get("/study/page", params=params)
    except httpx.HTTPError:
        raise ApiError(f"Cannot reach the backend at {BASE_URL}.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise ApiError(str(detail or r.reason_phrase), r.status_code)
    return r.content


def study_file_url(course: str, document: str) -> str:
    """Address the browser itself loads (video player), so it must be reachable from the browser."""
    from urllib.parse import urlencode
    return f"{BASE_URL}/study/file?{urlencode({'course': course, 'document': document})}"


def study_chat(body: dict) -> dict:
    """One full reply: {answer, sources, mode, language, retrieval_ms, total_s, provider}."""
    return _call("POST", "/study/chat", json={**body, "stream": False})


def study_chat_stream(body: dict, meta: dict):
    """Generator of answer pieces as the model writes them (for st.write_stream). `meta` is filled
    with the first line of the reply (sources, retrieval_ms) before the first piece is yielded."""
    try:
        with _http().stream("POST", "/study/chat", json={**body, "stream": True}) as r:
            if r.status_code >= 400:
                r.read()
                try:
                    detail = r.json().get("detail")
                except ValueError:
                    detail = r.text
                raise ApiError(str(detail or r.reason_phrase), r.status_code)
            import json
            header, got_header = "", False
            for piece in r.iter_text():
                if not got_header:
                    header += piece
                    if "\n" not in header:
                        continue
                    line, piece = header.split("\n", 1)
                    meta.update(json.loads(line))
                    got_header = True
                if piece:
                    yield piece
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise ApiError(f"Cannot reach the backend at {BASE_URL}. Start it with: uvicorn backend.main:app")
    except httpx.TimeoutException:
        raise ApiError("The backend did not answer in time.")


# --- teaching board (Q&A): corner video + highlighted page / whiteboard ---
def board(question: str, answer: str, course: str, lecture: Optional[int], sources: list[dict],
          provider: Optional[str] = None) -> dict:
    body = {"question": question, "answer": answer, "course": course, "lecture": lecture,
            "sources": sources, "provider": provider}
    return _call("POST", "/board", json=body)


def board_page_image(course: str, document: str, page: int, highlights: list[str]) -> bytes:
    """PNG of a PDF page / slide with the given lines highlighted."""
    params = [("course", course), ("document", document), ("page", page)] + [("hl", h) for h in highlights]
    try:
        r = _http().get("/board/page", params=params)
    except httpx.HTTPError:
        raise ApiError(f"Cannot reach the backend at {BASE_URL}.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise ApiError(str(detail or r.reason_phrase), r.status_code)
    return r.content


# --- explanation video (generated MP4 of the teaching board) ---
def start_lecture_video(question: str, course: str, board: dict) -> dict:
    return _call("POST", "/lecture-video", json={"question": question, "course": course, "board": board})


def lecture_video_status(job_id: str) -> dict:
    return _call("GET", f"/lecture-video/{job_id}")


def lecture_video_url(job_id: str) -> str:
    """Address the browser loads the MP4 from."""
    return f"{BASE_URL}/lecture-video/{job_id}/file"


def benchmark(sizes: Optional[list[int]] = None, provider: Optional[str] = None) -> list[dict]:
    return _call("POST", "/benchmark", json={"sizes": sizes, "provider": provider},
                 timeout=LONG)["rows"]


# --- feedback (usability / satisfaction) ---
def rate(rating_id: str, kind: str, up: bool, comment: str = "", context: Optional[dict] = None) -> dict:
    return _call("POST", "/feedback", json={"id": rating_id, "kind": kind, "up": up,
                                            "comment": comment, "context": context})


def submit_sus(responses: list[int], comment: str = "") -> dict:
    return _call("POST", "/survey", json={"responses": responses, "comment": comment})


def feedback_summary() -> dict:
    return _call("GET", "/feedback/summary")


# --- accounts (the token is passed per call: this module is shared by every browser session) ---
def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def register(student_no: str, password: str) -> dict:
    return _call("POST", "/auth/register", json={"student_no": student_no, "password": password})


def login(student_no: str, password: str) -> dict:
    return _call("POST", "/auth/login", json={"student_no": student_no, "password": password})


def logout(token: str) -> dict:
    return _call("POST", "/auth/logout", headers=_auth(token))


# --- practice history (My Results) ---
def create_test(token: str, course: str, lecture: Optional[int], difficulty: str, questions: list[dict]) -> str:
    body = {"course": course, "lecture": lecture, "difficulty": difficulty, "questions": questions}
    return _call("POST", "/tests", json=body, headers=_auth(token))["id"]


def save_attempt(token: str, attempt: dict) -> dict:
    return _call("POST", "/attempts", json=attempt, headers=_auth(token))


def history(token: str) -> dict:
    return _call("GET", "/history", headers=_auth(token))


def clear_history(token: str) -> dict:
    return _call("DELETE", "/history", headers=_auth(token))
