import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.config import get_config, resolve_path
from backend.indexing.indexer import correct_chunk, index_file
from backend.indexing.vectorstore import get_store
from backend.ingestion import EXTENSIONS

log = logging.getLogger(__name__)
_models = {"state": "not_loaded", "error": None}   # not_loaded | loading | ready | failed


def _preload_models() -> None:
    """Load (and on first run download) the embedding model and reranker before the first request,
    then run each once: the first inference on a GPU also initialises CUDA (seconds), which
    otherwise lands on the first student's question."""
    _models["state"] = "loading"
    try:
        from backend.indexing.embedder import embed_query
        embed_query("warm-up")
        if get_config()["retrieval"].get("use_reranker", True):
            from backend.retrieval.reranker import rerank_scores
            rerank_scores("warm-up", ["warm-up"])
        _models["state"] = "ready"
    except Exception as exc:  # noqa: BLE001
        log.warning("Model preload failed: %s", exc)
        _models.update(state="failed", error=str(exc))


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("RAG_PRELOAD", "1") != "0":
        threading.Thread(target=_preload_models, daemon=True).start()
    yield


app = FastAPI(title="Interactive Learning Assistant API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    course: str
    lecture: Optional[int] = None
    ts_range: Optional[list[float]] = None   # minutes [start, end] from the UI
    config: Optional[dict] = None


class GenerateRequest(BaseModel):
    course: str
    lecture: Optional[int] = None
    types: list[str]
    n: int = 5
    difficulty: str = "Medium"
    provider: Optional[str] = None


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    course: str
    lecture: Optional[int] = None
    provider: Optional[str] = None
    reference_answer: Optional[str] = None   # model answer of a generated question


class BenchmarkRequest(BaseModel):
    sizes: Optional[list[int]] = None
    provider: Optional[str] = None


class ChunkUpdate(BaseModel):
    text: str


class RatingRequest(BaseModel):
    id: Optional[str] = None                   # rated item; a new rating with the same id replaces it
    kind: str                                  # qa | evaluation
    up: bool
    comment: str = ""
    context: Optional[dict] = None             # question, course, provider, score ... (no raw documents)


class Credentials(BaseModel):
    student_no: str
    password: str


class TestRequest(BaseModel):
    course: str
    lecture: Optional[int] = None
    difficulty: Optional[str] = None
    questions: list[dict]


class AttemptRequest(BaseModel):
    test_id: str
    question_id: str
    student_answer: str
    is_correct: Optional[bool] = None          # choice questions
    score: Optional[int] = None                # open questions (rubric total)
    feedback: Optional[list[str]] = None
    dims: Optional[list[dict]] = None


class SusRequest(BaseModel):
    responses: list[int]
    comment: str = ""


class StudyRequest(BaseModel):
    course: str
    document: str
    mode: str = "ask"                          # ask | summary | questions | explain | concepts
    question: str = ""
    language: str = "en"                       # en | tr: language of the answer
    page: Optional[int] = None                 # page / slide the student is looking at
    time_ms: Optional[int] = None              # video: the moment the student is at
    section: Optional[str] = None              # docx: the section the student is reading
    history: Optional[list[dict]] = None       # earlier turns [{"role", "content"}], oldest first
    provider: Optional[str] = None
    stream: bool = False


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "provider": get_config()["llm"]["provider"],
            "models": _models["state"], "models_error": _models["error"]}


@app.get("/config")
def public_config():
    """Read-only settings the UI shows or uses as defaults (they live in config.yaml)."""
    cfg = get_config()
    return {"provider": cfg["llm"]["provider"],
            "embedding_model": cfg["indexing"]["embedding_model"],
            "chunk_size": cfg["indexing"]["chunk_size"],
            "chunk_overlap": cfg["indexing"]["chunk_overlap"],
            "whisper_model": cfg["ingestion"]["whisper_model"],
            "confidence_threshold": cfg["ingestion"]["confidence_threshold"],
            "reranker_model": cfg["retrieval"]["reranker_model"],
            "top_candidates": cfg["retrieval"]["top_candidates"],
            "top_context": cfg["retrieval"]["top_context"],
            "grounded_threshold": cfg["grounding"]["grounded_threshold"],
            "partial_threshold": cfg["grounding"]["partial_threshold"],
            "benchmark_sizes": cfg["benchmark"]["sizes"]}


@app.get("/courses")
def courses():
    return {"courses": get_store().list_courses()}


@app.get("/stats")
def stats():
    return get_store().stats()


@app.delete("/courses/{course}")
def delete_course(course: str, lecture: Optional[int] = None):
    return {"deleted": get_store().delete_course(course, lecture)}


@app.delete("/index")
def clear_index():
    get_store().clear()
    return {"cleared": True}


@app.get("/transcripts/low-confidence")
def low_confidence(threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
                   course: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    if threshold is None:
        threshold = get_config()["ingestion"]["confidence_threshold"]
    hits = get_store().low_confidence(threshold, course or None, limit)
    return {"segments": [{"id": h["id"], "text": h["text"], "document": h["metadata"]["document"],
                          "course": h["metadata"]["course"], "lecture": h["metadata"]["lecture"],
                          "start_ms": h["metadata"]["timestamp_start_ms"],
                          "end_ms": h["metadata"]["timestamp_end_ms"],
                          "confidence": h["metadata"]["confidence"]} for h in hits]}


@app.put("/chunks/{chunk_id}")
def update_chunk(chunk_id: str, body: ChunkUpdate):
    try:
        hit = correct_chunk(chunk_id, body.text)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No chunk with id {chunk_id}")
    except Exception as exc:  # noqa: BLE001
        raise _http(exc)
    return {"id": hit["id"], "text": hit["text"], "confidence": hit["metadata"]["confidence"]}


@app.get("/documents")
def documents():
    return {"documents": get_store().list_documents()}


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...), course: str = Form(...), lecture: int = Form(...)):
    upload_dir = resolve_path("data/uploads") / course
    upload_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for f in files:
        name = Path(f.filename).name                      # strip any path components
        if Path(name).suffix.lower() not in EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {name}")
        dest = upload_dir / name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            records.append(index_file(dest, course, lecture))
        except Exception as exc:  # noqa: BLE001
            raise _http(exc)
    return {"records": records}


@app.post("/ask")
def ask(req: AskRequest):
    from backend.tasks.qa import answer_question
    window = (int(req.ts_range[0] * 60_000), int(req.ts_range[1] * 60_000)) if req.ts_range else None
    try:
        result = answer_question(req.question, req.course, req.lecture, window, req.config)
    except Exception as exc:  # noqa: BLE001
        raise _http(exc)
    result.pop("contexts", None)
    return result


# --- Interactive Learning: one open document + a study-buddy chat limited to it ---
def _study_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (LookupError, FileNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    return _http(exc)


@app.get("/study/document")
def study_document(course: str, document: str):
    from backend.tasks.study import outline
    try:
        return outline(course, document)
    except Exception as exc:  # noqa: BLE001
        raise _study_error(exc)


@app.get("/study/page")
def study_page(course: str, document: str, page: int = Query(1, ge=1),
               zoom: float = Query(1.5, ge=0.5, le=3.0)):
    from fastapi.responses import Response
    from backend.tasks.study import render_page
    try:
        png = render_page(course, document, page, zoom)
    except Exception as exc:  # noqa: BLE001
        raise _study_error(exc)
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/study/file")
def study_file(course: str, document: str):
    """The original upload, shown inline (used by the video player)."""
    from fastapi.responses import FileResponse
    from backend.tasks.study import upload_path
    try:
        path = upload_path(course, document)
    except Exception as exc:  # noqa: BLE001
        raise _study_error(exc)
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@app.post("/study/chat")
def study_chat(req: StudyRequest):
    """With stream=true the body is one JSON line (sources, timings) followed by the answer text
    as it is generated."""
    import json
    from fastapi.responses import StreamingResponse
    from backend.tasks.study import study_answer, study_stream
    kwargs = req.model_dump(exclude={"provider", "stream"})
    try:
        if not req.stream:
            return study_answer(provider=req.provider, **kwargs)
        meta, pieces = study_stream(provider=req.provider, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _study_error(exc)

    def body():
        yield json.dumps(meta, ensure_ascii=False) + "\n"
        try:
            yield from pieces
        except Exception as exc:  # noqa: BLE001  (headers are sent: report it in the text)
            log.warning("Study chat failed: %s", exc)
            yield f"\n\n⚠️ The language model is not reachable right now ({exc})."
    return StreamingResponse(body(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate")
def generate(req: GenerateRequest):
    from backend.tasks.question_gen import generate_questions
    try:
        return {"questions": generate_questions(req.course, req.lecture, req.types, req.n,
                                                req.difficulty, req.provider)}
    except Exception as exc:  # noqa: BLE001
        raise _http(exc)


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    from backend.tasks.evaluator import evaluate_answer
    try:
        return evaluate_answer(req.question, req.answer, req.course, req.lecture, req.provider,
                               req.reference_answer)
    except Exception as exc:  # noqa: BLE001
        raise _http(exc)


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    from backend.evaluation.benchmark import run_benchmark
    try:
        df = run_benchmark(req.sizes, provider=req.provider)
    except Exception as exc:  # noqa: BLE001
        raise _http(exc)
    return {"rows": df.to_dict("records")}


@app.post("/feedback")
def feedback(req: RatingRequest):
    from backend.evaluation.feedback import add_rating
    try:
        add_rating(req.kind, req.up, req.comment, req.context, req.id)
    except ValueError as exc:
        raise _http(exc)
    return {"saved": True}


@app.post("/survey")
def survey(req: SusRequest):
    from backend.evaluation.feedback import add_sus, sus_grade
    try:
        score = add_sus(req.responses, req.comment)
    except ValueError as exc:
        raise _http(exc)
    return {"score": score, "grade": sus_grade(score)}


@app.get("/feedback/summary")
def feedback_summary():
    from backend.evaluation.feedback import summary
    return summary()


# --- student accounts (practice history is per student) ---
def current_student(authorization: Optional[str] = Header(None)) -> str:
    from backend.users import student_for
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    student = student_for(token)
    if not student:
        raise HTTPException(status_code=401, detail="Please log in again")
    return student


@app.post("/auth/register")
def register(body: Credentials):
    from backend.users import register as create
    try:
        return {"token": create(body.student_no, body.password), "student_no": body.student_no.strip()}
    except ValueError as exc:
        raise _http(exc)


@app.post("/auth/login")
def login(body: Credentials):
    from backend.users import login as check
    try:
        return {"token": check(body.student_no, body.password), "student_no": body.student_no.strip()}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.get("/auth/me")
def me(student: str = Depends(current_student)):
    return {"student_no": student}


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    from backend.users import logout as end
    if authorization:
        end(authorization.removeprefix("Bearer ").strip())
    return {"logged_out": True}


# --- practice history (My Results) ---
@app.post("/tests")
def create_test(req: TestRequest, student: str = Depends(current_student)):
    from backend.evaluation.history import create_test as save
    try:
        return {"id": save(student, req.course, req.lecture, req.difficulty, req.questions)}
    except ValueError as exc:
        raise _http(exc)


@app.post("/attempts")
def add_attempt(req: AttemptRequest, student: str = Depends(current_student)):
    from backend.evaluation.history import add_attempt as save
    try:
        return {"id": save(student, req.test_id, req.question_id, req.student_answer,
                           req.is_correct, req.score, req.feedback, req.dims)}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _http(exc)


@app.get("/history")
def history(student: str = Depends(current_student)):
    from backend.evaluation.history import list_tests
    return list_tests(student)


@app.delete("/history")
def clear_history(student: str = Depends(current_student)):
    from backend.evaluation.history import delete_history
    return {"deleted_tests": delete_history(student)}
