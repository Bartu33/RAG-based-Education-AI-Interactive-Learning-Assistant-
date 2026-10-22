"""A lecture video a student uploads to study with: transcribed and indexed in the background.

The transcript is indexed in steps (every `step_s` seconds of audio), so the video can be watched
and the study buddy can answer about the first minutes while the rest is still being transcribed.
Each step re-chunks the transcript so far and stores only the chunks that are new or changed;
chunk ids are the usual position-based ids, so the result is the same as a one-shot /ingest.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from backend.config import get_config
from backend.indexing.chunker import chunk_segments
from backend.indexing.embedder import embed_texts
from backend.indexing.vectorstore import VectorStore, chunk_id, get_store

log = logging.getLogger(__name__)
DEFAULTS = {
    "beam_size": 1,    # greedy decoding: about twice as fast as beam 5, slightly less accurate
    "step_s": 90,      # index the transcript every this many seconds of audio
}
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_WORKER = threading.Semaphore(1)   # one transcription at a time (CPU / GPU heavy)


def upload_cfg() -> dict:
    return {**DEFAULTS, **(get_config().get("video_upload") or {})}


def _public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "course", "lecture", "document", "state", "done_ms",
                                "duration_ms", "chunks", "error")}


class _Indexer:
    """Stores the transcript so far: chunks that did not change since the last step are kept."""

    def __init__(self, course: str, lecture: int, document: str, store: VectorStore):
        self.course, self.lecture, self.document, self.store = course, lecture, document, store
        self.stored = 0   # chunks [0, stored - 1) are final; the last one may still grow

    def __call__(self, segments: list) -> int:
        chunks = chunk_segments(segments)
        start = max(0, self.stored - 1)
        new = chunks[start:]
        if new:
            ids = [chunk_id(self.course, self.lecture, self.document, i)
                   for i in range(start, len(chunks))]
            self.store.add(new, embed_texts([c.text for c in new]), ids=ids)
        self.stored = len(chunks)
        return len(chunks)


def run(job: dict, path: Path, store: Optional[VectorStore] = None, transcribe=None) -> None:
    """Transcribe and index (blocking). `transcribe(path, beam_size)` -> (duration_ms, segments)."""
    from backend.ingestion.video_stt import iter_transcript
    cfg = upload_cfg()
    store = store or get_store()
    transcribe = transcribe or iter_transcript
    with _WORKER:
        try:
            job["state"] = "transcribing"
            store.delete_document(job["document"], job["course"])   # an upload replaces the old one
            index = _Indexer(job["course"], job["lecture"], job["document"], store)
            duration, pieces = transcribe(path, int(cfg["beam_size"]))
            job["duration_ms"] = duration
            segments, next_step = [], int(cfg["step_s"]) * 1000
            for seg in pieces:
                seg.course, seg.lecture = job["course"], job["lecture"]
                segments.append(seg)
                job["done_ms"] = seg.timestamp_end_ms
                if seg.timestamp_end_ms >= next_step:
                    job["chunks"] = index(segments)
                    next_step = seg.timestamp_end_ms + int(cfg["step_s"]) * 1000
            if not segments:
                raise ValueError("No speech was found in this video.")
            job["chunks"] = index(segments)
            job.update(state="done", done_ms=duration or job["done_ms"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Video %s failed: %s", job["document"], exc)
            job.update(state="failed", error=str(exc))
        finally:
            job["finished"] = time.time()


def start(path, course: str, lecture: int) -> dict:
    path = Path(path)
    job = {"id": uuid.uuid4().hex[:12], "course": course, "lecture": int(lecture),
           "document": path.name, "state": "queued", "done_ms": 0, "duration_ms": None,
           "chunks": 0, "error": None, "finished": None}
    with _LOCK:
        _JOBS[job["id"]] = job
    threading.Thread(target=run, args=(job, path), daemon=True, name=f"video-{job['id']}").start()
    return _public(job)


def status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if not job:
        raise LookupError("No such video job (the backend may have restarted).")
    return _public(job)


def status_for(course: str, document: str) -> Optional[dict]:
    """The latest job of a video, if one ran since the backend started."""
    jobs = [j for j in _JOBS.values() if j["course"] == course and j["document"] == document]
    return _public(jobs[-1]) if jobs else None
