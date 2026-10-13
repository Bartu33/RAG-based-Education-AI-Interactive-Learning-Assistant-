# API Contract

Backend: FastAPI (`backend/main.py`), default `http://localhost:8000`, interactive docs at `/docs`.
This file documents what the backend **returns today**. The Streamlit UI must only talk to the
backend through these endpoints (see [Rules](#rules)). When an endpoint changes, update this file in
the same PR.

## Rules

1. **The UI never imports `backend.*`.** ChromaDB must not be opened by two processes at the same
   time, so all access goes through the API (`rag_ui/core/api_client.py`).
2. Keys below are the source of truth. The UI reaches the API at `RAG_API_URL`
   (default `http://localhost:8000`).
3. Errors: `400` = bad input (`ValueError`, e.g. no indexed material), `503` = backend/provider
   failure (missing API key, LLM down), `422` = request validation. Body: `{"detail": "<message>"}`.
4. Optional fields may be `null`: `page` (non-pdf/pptx), `start_ms` / `end_ms` (non-video),
   `faithfulness` (when `qa.score_faithfulness: false`), `provider` (when nothing was retrieved).

## Endpoints

### `GET /health`
```json
{"status": "ok", "provider": "groq", "models": "ready", "models_error": null}
```
`models`: `not_loaded | loading | ready | failed`. On startup the backend loads (on the first run:
downloads) the embedding model and the reranker in a background thread so the first question does
not time out. Set `RAG_PRELOAD=0` to skip this.

### `GET /config`
Read-only settings from `config.yaml` that the UI shows or uses as defaults.
```json
{"provider": "groq", "embedding_model": "BAAI/bge-m3", "chunk_size": 500, "chunk_overlap": 75,
 "whisper_model": "small", "confidence_threshold": 0.6, "reranker_model": "BAAI/bge-reranker-v2-m3",
 "top_candidates": 20, "top_context": 4, "grounded_threshold": 0.85, "partial_threshold": 0.5,
 "benchmark_sizes": [1000, 10000, 50000, 100000]}
```

### `GET /courses`
```json
{"courses": [{"course": "CENG336", "lectures": [1, 2], "documents": 3, "chunks": 210}]}
```

### `GET /stats`
```json
{"collection": "course_chunks", "chunks": 210, "by_type": {"pdf": 180, "video": 30}, "courses": 1,
 "embedding_model": "BAAI/bge-m3", "hnsw": {"M": 16, "construction_ef": 100, "search_ef": 100}}
```

### `DELETE /courses/{course}?lecture=1`
`lecture` is optional (omit it to remove the whole course). → `{"deleted": 93}` (chunks removed).

### `DELETE /index`
Empties the collection. → `{"cleared": true}`

### `GET /transcripts/low-confidence?threshold=0.6&course=CENG336&limit=50`
All parameters optional; `threshold` defaults to `ingestion.confidence_threshold`. Lowest confidence first.
```json
{"segments": [{"id": "9f2c…", "text": "the nyquist rate …", "document": "Lecture01.mp4",
  "course": "CENG336", "lecture": 1, "start_ms": 60000, "end_ms": 90000, "confidence": 0.41}]}
```

### `PUT /chunks/{id}`
Instructor correction: the text is re-embedded and the chunk is marked corrected (video confidence → 1.0).
```json
{"text": "the Nyquist rate …"}
```
→ `{"id": "9f2c…", "text": "the Nyquist rate …", "confidence": 1.0}`; unknown id → `404`, empty text → `400`.

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

## Not exposed yet (backend functions exist)

| Function | Suggested endpoint | UI use |
|---|---|---|
| `index_file(..., on_progress=fn)` | background `/ingest` + `GET /jobs/{id}` | per-stage progress bar (the UI now shows per-file progress) |
| `retrieve(...)` → `timings`, `mode`, per-chunk `similarity`, `rerank`, `duplicates` | `POST /retrieve` (debug) | "context sent externally" panel |
