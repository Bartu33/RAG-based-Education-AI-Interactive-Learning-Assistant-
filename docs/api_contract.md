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
 "config": {"provider": "Groq (GPT-OSS 120B)", "top_k_candidates": 20, "top_k_context": 4}}
```
`lecture`, `ts_range` (minutes the student is watching), `config` are optional.
`config.provider` accepts the UI label; it is mapped with `resolve_provider_name`.

```json
{"answer": "...[1][2]",
 "found": true,
 "beyond": "General background that is not in the material ... (or null)",
 "faithfulness": 0.92,
 "sources": [
   {"type": "pdf", "document": "Chapter_1.pdf", "page": 39, "start_ms": null, "end_ms": null,
    "score": 0.927, "text": "first 400 characters ..."}
 ],
 "retrieval_ms": 140, "total_s": 2.3, "tokens_sent": 1850,
 "candidates": 20, "filtered_from": 93, "provider": "groq"}
```
`found` is `false` when the material does not cover the question: either nothing was retrieved
(`sources` = []) or the model replied with the `[NO_INFO]` marker the QA prompt asks for (the marker
is removed; `sources` then lists the closest passages that were checked). In both cases
`faithfulness` = 1.0 (no claim was made) and no faithfulness LLM call is spent.

`beyond` (`qa.general_knowledge: true`): an optional part the model writes after the
`[BEYOND]` marker — general subject knowledge and reasoning that is **not** in the course
material (background, an example, or the answer itself when `found` is `false`). It is removed from
`answer`, shown in a separate light-blue box, and never scored by the faithfulness check, so the
grounding / hallucination metrics only cover what the material supports. `null` when the model
added nothing or the option is off.

### `POST /generate`
```json
{"course": "CENG336", "lecture": 1,
 "counts": {"Multiple Choice": 5, "True/False": 1, "Short Answer": 1, "Conceptual/Computational": 1},
 "topic": "sampling theorem", "difficulty": "Medium", "provider": null, "exam_ids": null}
```
`counts`: questions per type, 0–10 each, at most 20 in total (otherwise `400`); the questions come
back grouped in this order. `topic` (optional): the excerpts are the ones the Q&A search finds for
it (`400` if nothing matches); without it, a random sample of the course / lecture. The older form
`"types": [...], "n": 5` (types cycled) still works when `counts` is omitted. `exam_ids` (optional):
past exams of the same course (`/exams`); the questions are then written in the instructor's style,
with up to 8 of the exams' questions as examples, and, without a `topic`, from the excerpts about
the topics the exams ask about. Such questions carry `"style_ref": "Midterm 2023: <example>"`.
```json
{"questions": [
  {"id": "a1b2c3d4", "type": "Multiple Choice", "difficulty": "Medium",
   "question": "...", "options": ["...", "...", "...", "..."], "correct_answer": "...",
   "explanation": "...", "source": "Chapter_1.pdf, p. 39"}
]}
```
`options` is `null` for non-MCQ. Each LLM item is checked against the `format` it states (so a
reordered reply loses nothing); invalid items and extras beyond the requested count per type are
dropped, so fewer questions than requested may come back.

### Past exams: `GET /exams?course=CENG336`, `POST /exams`, `DELETE /exams/{id}`
`POST /exams` is multipart/form-data: `course`, `name` (optional; the file name by default), and
either `file` (pdf / docx / pptx) or `text` (pasted questions, e.g. of a scanned exam). The LLM reads
the first ~12,000 characters once; `400` when the file has no text or no question is recognised.
Exams are stored in SQLite, not in the vector index: Q&A and the study buddy never cite them.
```json
{"id": "3f2a…", "course": "CENG336", "name": "Midterm 2023", "created": "2026-10-21T10:00:00+00:00",
 "truncated": false, "summary": "Mostly calculations; every answer must be justified.",
 "style_points": ["gives a signal and asks for its period"], "topics": ["periodicity", "convolution"],
 "mix": {"mcq": 0, "tf": 1, "short": 0, "concept": 2},
 "questions": [{"text": "Is x(n)=cos(0.3πn) periodic? Justify.", "topic": "periodicity", "format": "concept"}]}
```
`GET` returns `{"exams": [...]}` in this shape; `DELETE` returns `{"deleted": true}` or `404`.

### `POST /evaluate`
```json
{"question": "...", "answer": "student's free text", "course": "CENG336", "lecture": 1, "provider": null,
 "reference_answer": "model answer of a generated question (optional)"}
```
Multiple-choice and true/false questions are checked in the UI against `correct_answer`; only
open-ended answers come here (from the Question Generation page, with the model answer as
`reference_answer`).
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

### `POST /feedback`
A 👍/👎 on one Q&A answer (`kind` = `qa`) or one rubric evaluation (`evaluation`). A later rating
with the same `id` (changed thumb, added comment) replaces the earlier one in the summary.
```json
{"id": "9ecc414edc91", "kind": "qa", "up": false, "comment": "wrong page",
 "context": {"question": "...", "course": "CENG336", "found": true, "faithfulness": 0.9}}
```
→ `{"saved": true}`; unknown `kind` → `400`.

### `POST /survey`
System Usability Scale: 10 answers from 1 (strongly disagree) to 5 (strongly agree).
```json
{"responses": [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], "comment": ""}
```
→ `{"score": 75.0, "grade": "Good"}` (0–100; grades after Bangor et al., 2009); wrong length/value → `400`.

### `GET /feedback/summary`
```json
{"ratings": {"qa": {"count": 12, "up": 10, "down": 2, "positive_pct": 83.3},
             "evaluation": {"count": 5, "up": 4, "down": 1, "positive_pct": 80.0}},
 "sus": {"count": 8, "mean": 76.9, "stdev": 9.4, "grade": "Good"},
 "comments": [{"kind": "qa", "up": false, "comment": "wrong page", "time": "..."}]}
```
Stored as JSON lines in `feedback.path` (`data/feedback/`, git-ignored because it holds student text).

### Accounts: `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
```json
{"student_no": "220201032", "password": "secret1"}
```
→ `{"token": "...", "student_no": "220201032"}`. Register: 3–32 letters/digits/`._-`, password ≥ 6
characters, an existing number → `400`. Wrong number or password → `401`. Send the token as
`Authorization: Bearer <token>` to the history endpoints below; a missing/expired token → `401`
(sessions last 30 days, logout deletes the token). Passwords are stored as salted PBKDF2-SHA256 hashes.

### `POST /tests` (logged in)
One "Generate questions" run. The questions are stored with the test, so unanswered ones can be shown.
```json
{"course": "CENG336", "lecture": 1, "difficulty": "Medium", "questions": [{"id": "a1b2c3d4", "type": "...",
 "question": "...", "options": ["..."], "correct_answer": "...", "explanation": "...", "source": "..."}]}
```
→ `{"id": "<test id>"}`.

### `POST /attempts` (logged in)
```json
{"test_id": "...", "question_id": "a1b2c3d4", "student_answer": "B) ...", "is_correct": false,
 "score": null, "feedback": null, "dims": null}
```
`is_correct` for choice questions, `score` (+ `feedback`, `dims`) for open ones. Someone else's or an
unknown test → `404`; a question not in the test or an empty answer → `400`.

### `GET /history` / `DELETE /history` (logged in)
Tests newest first, each with its questions and the latest attempt per question (`null` = not answered).
```json
{"tests": [{"id": "...", "number": 3, "created": "2026-10-16T14:20:03+00:00", "course": "CENG336",
            "lecture": 1, "difficulty": "Medium",
            "questions": [{"id": "...", "question": "...", "correct_answer": "...",
                           "attempt": {"student_answer": "...", "is_correct": false, "score": null,
                                       "feedback": null, "tries": 2, "time": "..."}}],
            "summary": {"questions": 5, "answered": 4, "choice": {"count": 3, "correct": 2},
                        "open": {"count": 1, "average": 72.0}}}],
 "summary": {"tests": 3, "questions": 15, "answered": 12, "choice": {"...": "..."}, "open": {"...": "..."}}}
```
`DELETE` → `{"deleted_tests": 3}`. Everything lives in SQLite at `database.path` (`data/app.db`, git-ignored).

### Interactive Learning (one open document + study-buddy chat)
The chat only searches the chunks of the open document: they are read once and kept in memory
until the next write to the store, so a question costs one local embedding + a dot product over
those chunks. Reranker and faithfulness check are off by default (`study:` in `config.yaml`).

`GET /study/document?course=CENG336&document=Chapter_1.pdf` → what the viewer shows. 404 if not indexed.
```json
{"document": "Chapter_1.pdf", "course": "CENG336", "lecture": 1, "type": "pdf", "chunks": 93,
 "pages": 92, "has_file": true,
 "units": [{"label": "Page 1", "page": 1, "section": null, "start_ms": null, "end_ms": null, "text": "..."}]}
```
Units are pages (pdf), slides (pptx), sections (docx) or transcript parts (video), in reading order.

`GET /study/page?course=..&document=..&page=3&zoom=1.6` → `image/png` of one PDF page (PyMuPDF).
`GET /study/file?course=..&document=..` → the original upload, inline (video player, "open original").
Both return 404 when the original file is not in `data/uploads/<course>/`, 400 for invalid names.

`POST /study/chat`
```json
{"course": "CENG336", "document": "Chapter_1.pdf", "mode": "ask", "question": "What is a protocol?",
 "language": "tr", "page": 3, "time_ms": null, "section": null,
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "provider": null, "stream": true}
```
`mode`: `ask` (needs `question`) · `summary` · `explain` (uses `page` / `time_ms` / `section`)
· `concepts`. `language`: `en` | `tr` (language of the answer). `page`/`time_ms`/`section` = where the
student is; in `ask` mode chunks there get a small bonus (`study.page_boost`).

- `stream: false` → `{"answer", "beyond", "sources", "mode", "language", "retrieval_ms", "total_s", "provider"}`
- `stream: true` → `text/plain`: the first line is JSON `{"sources", "mode", "language", "retrieval_ms"}`,
  everything after it is the answer as it is generated. The "beyond the document" part follows a
  `[BEYOND]` marker in the text (`study.general_knowledge: true`); the UI splits it off.

`sources`: `[{"n": 1, "label": "Page 3", "page": 3, "section": null, "start_ms": null, "end_ms": null,
"score": 0.61, "text": "first 300 characters"}]` (`n` matches the `[1]` citations). Errors before the
answer starts: 404 unknown document, 400 empty question / unknown mode / nothing indexed at that page.

## Not exposed yet (backend functions exist)

| Function | Suggested endpoint | UI use |
|---|---|---|
| `index_file(..., on_progress=fn)` | background `/ingest` + `GET /jobs/{id}` | per-stage progress bar (the UI now shows per-file progress) |
| `retrieve(...)` → `timings`, `mode`, per-chunk `similarity`, `rerank`, `duplicates` | `POST /retrieve` (debug) | "context sent externally" panel |
