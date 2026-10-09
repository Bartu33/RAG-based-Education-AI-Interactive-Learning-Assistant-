from pathlib import Path

from backend.config import get_config
from backend.indexing.chunker import chunk_segments
from backend.indexing.embedder import embed_texts
from backend.indexing.vectorstore import VectorStore, get_store
from backend.ingestion import load_file, source_type_of


def index_file(path, course: str, lecture: int, store: VectorStore | None = None) -> dict:
    """Extract -> chunk -> embed (local) -> store. Returns one record for the UI."""
    store = store or get_store()
    path = Path(path)
    segments = load_file(path)
    for s in segments:
        s.course, s.lecture = course, int(lecture)
    chunks = chunk_segments(segments)

    store.delete_document(path.name, course)  # re-indexing replaces old chunks
    if chunks:
        store.add(chunks, embed_texts([c.text for c in chunks]))

    threshold = get_config()["ingestion"]["confidence_threshold"]
    low = sum(1 for c in chunks if c.confidence is not None and c.confidence < threshold)
    return {"document": path.name, "course": course, "lecture": int(lecture),
            "type": source_type_of(path), "chunks": len(chunks),
            "low_confidence_segments": low}