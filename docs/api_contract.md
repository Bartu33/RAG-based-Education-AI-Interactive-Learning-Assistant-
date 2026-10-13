# API Contract

Backend: FastAPI (`backend/main.py`), default `http://localhost:8000`, interactive docs at `/docs`.
This file documents what the backend **returns today**. The Streamlit UI must only talk to the
backend through these endpoints (see [Rules](#rules)). When an endpoint changes, update this file in
the same PR.

## Rules

1. **The UI never imports `backend.*`.** ChromaDB must not be opened by two processes at the same
   time, so all access goes through the API (`rag_ui/core/api_client.py`).
2. Keys below are the source of truth. `rag_ui/core/mock_backend.py` uses some different keys; see
   [Mock vs. backend](#mock-vs-backend).
3. Errors: `400` = bad input (`ValueError`, e.g. no indexed material), `503` = backend/provider
   failure (missing API key, LLM down), `422` = request validation. Body: `{"detail": "<message>"}`.
4. Optional fields may be `null`: `page` (non-pdf/pptx), `start_ms` / `end_ms` (non-video),
   `faithfulness` (when `qa.score_faithfulness: false`), `provider` (when nothing was retrieved).

## Endpoints

### `GET /health`
```json
{"status": "ok", "provider": "groq"}
```

### `GET /documents`
```json
{"documents": [
  {"document": "Chapter_1.pdf", "course": "CENG336", "lecture": 1, "type": "pdf", "chunks": 93}
]}
```

### `POST /ingest` (multipart/form-data)
Fields: `files` (one or more files: `.pdf .pptx .docx .mp4 .mkv .avi`), `course` (str), `lecture` (int).
Re-uploading a file with the same name in the same course replaces its old chunks.

```bash
curl -X POST http://localhost:8000/ingest -F "files=@Chapter_1.pdf" -F course=CENG336 -F lecture=1
```
> Swagger UI (OpenAPI 3.1) shows `files` as a text box and sends the path as a string → `422`.
> Use curl, the UI, or `python -m scripts.ingest_local` for now.

```json
{"records": [
  {"document": "Chapter_1.pdf", "course": "CENG336", "lecture": 1, "type": "pdf", "chunks": 93,
   "low_confidence_segments": 0,
   "timings_ms": {"extract": 520, "chunk": 40, "embed": 2600, "store": 93}}
]}
```
The request is synchronous: a long video can exceed client timeouts (planned: background task +
progress, see `index_file(on_progress=...)`).

### `POST /ask`
```json
{"question": "Paket anahtarlama nedir?", "course": "CENG336", "lecture": 1,
 "ts_range": [12, 14],
 "config": {"provider": "Groq (Llama 3)", "top_k_candidates": 20, "top_k_context": 4}}
```
`lecture`, `ts_range` (minutes the student is watching), `config` are optional.
`config.provider` accepts the UI label; it is mapped with `resolve_provider_name`.

```json
{"answer": "...[1][2]",
 "faithfulness": 0.92,
 "sources": [
   {"type": "pdf", "document": "Chapter_1.pdf", "page": 39, "start_ms": null, "end_ms": null,
    "score": 0.927, "text": "first 400 characters ..."}
 ],
 "retrieval_ms": 140, "total_s": 2.3, "tokens_sent": 1850,
 "candidates": 20, "filtered_from": 93, "provider": "groq"}
```
No chunk found → `answer` = "I could not find this in the approved course materials.",
`faithfulness` = 1.0, `sources` = [].

### `POST /generate`
```json
{"course": "CENG336", "lecture": 1, "types": ["Multiple Choice", "True/False"], "n": 5,
 "difficulty": "Medium", "provider": null}
```
```json
{"questions": [
  {"id": "a1b2c3d4", "type": "Multiple Choice", "difficulty": "Medium",
   "question": "...", "options": ["...", "...", "...", "..."], "correct_answer": "...",
   "explanation": "...", "source": "Chapter_1.pdf, p. 39"}
]}
```
`options` is `null` for non-MCQ. Invalid LLM items are dropped, so fewer than `n` may come back.

### `POST /evaluate`
```json
{"question": "...", "answer": "student's free text", "course": "CENG336", "lecture": 1, "provider": null}
```
```json
{"total": 72,
 "dims": [
   {"name": "Factual accuracy", "weight": 0.4, "score": 0.8},
   {"name": "Key concept coverage", "weight": 0.3, "score": 0.67},
   {"name": "Course terminology", "weight": 0.2, "score": 0.61},
   {"name": "Completeness", "weight": 0.1, "score": 0.7}
 ],
 "feedback": ["...", "..."], "source": "Chapter_1.pdf, p. 39", "follow_up_question": "..."}
```
`course` is required: the evaluation retrieves the course context first.

### `POST /benchmark`
```json
{"sizes": [1000, 10000], "provider": null}
```
```json
{"rows": [
  {"Corpus (chunks)": 1000, "Retrieval (ms)": 45.0, "Answer (s)": 2.1, "RAGAs Faithfulness": 0.91,
   "Hallucination (%)": 4.0, "Intrinsic (%)": 1.0, "Extrinsic (%)": 3.0}
]}
```

## Mock vs. backend

Differences the UI must fix when it switches from `mock_backend.py` to the API:

| Endpoint | Mock key | Backend key |
|---|---|---|
| `/ingest` records | `Document`, `Course`, `Week`, `Type`, `Chunks`, `Low-confidence segments` | `document`, `course`, `lecture`, `type`, `chunks`, `low_confidence_segments` |
| `/evaluate` | `dimensions` | `dims` |
| `/evaluate` | (UI sends `course=""`) | `course` must be a real course |
| `/benchmark` | `Response (s)` | `Answer (s)` |
| `/benchmark` | intrinsic/extrinsic hard-coded in the view (3.2 % / 7.8 %) | `Intrinsic (%)`, `Extrinsic (%)` per row |
| `/ask` | `faithfulness` always a number | may be `null` → `label_for(None)` must not crash |
| courses | `COURSES` dict in the mock | no endpoint yet (store has `list_courses()`) |

## Not exposed yet (backend functions exist)

Ready in `backend/indexing/` for whoever adds the endpoints:

| Function | Suggested endpoint | UI use |
|---|---|---|
| `get_store().list_courses()` | `GET /courses` | course dropdowns instead of the hard-coded list |
| `get_store().stats()` | `GET /stats` | home page counters |
| `get_store().delete_course(course, lecture=None)` / `clear()` | `DELETE /courses/{course}` / `DELETE /index` | "Clear entire index" |
| `get_store().low_confidence(threshold, course)` | `GET /transcripts/low-confidence` | "Edit transcript" list |
| `indexer.correct_chunk(chunk_id, text)` | `PUT /chunks/{id}` | "Save correction" |
| `index_file(..., on_progress=fn)` | background `/ingest` + `GET /jobs/{id}` | progress bar |
| `retrieve(...)` → `timings`, `mode`, per-chunk `similarity`, `rerank`, `duplicates` | `POST /retrieve` (debug) | "context sent externally" panel |

Settings page: the embedding model choice is not sent to the backend. Changing the embedding
model requires re-indexing; the store refuses to mix models (`EmbeddingModelMismatch`), so the
option should be removed or show a "re-index required" warning.
