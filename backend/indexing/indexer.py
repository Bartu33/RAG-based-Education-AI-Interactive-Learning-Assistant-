"""Extract -> chunk -> embed (local) -> store.

`on_progress(stage, fraction)` lets a caller (e.g. a FastAPI background task) report progress:
stages are "extract", "chunk", "embed", "store", "done"; fraction is 0..1 for the whole file.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from backend.config import get_config
from backend.indexing.chunker import chunk_segments
from backend.indexing.embedder import embed_texts
from backend.indexing.vectorstore import VectorStore, get_store
from backend.ingestion import load_file, source_type_of

ProgressFn = Callable[[str, float], None]

# rough share of the total time each stage takes (video: STT dominates the extract stage)
_WEIGHTS = {"extract": 0.35, "chunk": 0.05, "embed": 0.5, "store": 0.1}
_EMBED_BATCH = 256  # chunks per progress update while embedding


def _noop(stage: str, fraction: float) -> None:
    pass


def index_file(path, course: str, lecture: int, store: VectorStore | None = None,
               on_progress: Optional[ProgressFn] = None) -> dict:
    """Index one file. Returns one record for the UI (same keys as before + `timings_ms`)."""
    store = store or get_store()
    report = on_progress or _noop
    path = Path(path)
    timings: dict[str, int] = {}
    done = 0.0

    def stage(name: str, t0: float) -> None:
        nonlocal done
        timings[name] = int((perf_counter() - t0) * 1000)
        done += _WEIGHTS[name]
        report(name, min(done, 1.0))

    t = perf_counter()
    report("extract", 0.0)
    segments = load_file(path)
    for s in segments:
        s.course, s.lecture = course, int(lecture)
    stage("extract", t)

    t = perf_counter()
    chunks = chunk_segments(segments)
    stage("chunk", t)

    t = perf_counter()
    vectors = []
    texts = [c.text for c in chunks]
    for i in range(0, len(texts), _EMBED_BATCH):
        vectors.extend(embed_texts(texts[i:i + _EMBED_BATCH]))
        report("embed", done + _WEIGHTS["embed"] * min(1.0, (i + _EMBED_BATCH) / len(texts)))
    stage("embed", t)

    t = perf_counter()
    store.delete_document(path.name, course)  # re-indexing replaces old chunks
    if chunks:
        store.add(chunks, vectors)
    stage("store", t)
    report("done", 1.0)

    threshold = get_config()["ingestion"]["confidence_threshold"]
    low = sum(1 for c in chunks if c.confidence is not None and c.confidence < threshold)
    return {"document": path.name, "course": course, "lecture": int(lecture),
            "type": source_type_of(path), "chunks": len(chunks),
            "low_confidence_segments": low, "timings_ms": timings}


def correct_chunk(chunk_id: str, text: str, store: VectorStore | None = None) -> dict:
    """Instructor fix of one chunk (e.g. a low-confidence transcript segment): re-embed and save.
    The chunk keeps its document/page/timestamps; it is marked `corrected` (video confidence -> 1.0)."""
    store = store or get_store()
    text = text.strip()
    if not text:
        raise ValueError("Corrected text is empty")
    store.update_text(chunk_id, text, embed_texts([text])[0])
    return store.get_by_ids([chunk_id])[0]
