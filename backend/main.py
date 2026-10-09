import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import get_config, resolve_path
from backend.indexing.indexer import index_file
from backend.indexing.vectorstore import get_store
from backend.ingestion import EXTENSIONS

app = FastAPI(title="Interactive Learning Assistant API")


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


class BenchmarkRequest(BaseModel):
    sizes: Optional[list[int]] = None
    provider: Optional[str] = None


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "provider": get_config()["llm"]["provider"]}


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
        return evaluate_answer(req.question, req.answer, req.course, req.lecture, req.provider)
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