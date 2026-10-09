from backend.indexing.chunker import chunk_segments
from backend.schemas import Segment


def test_text_chunks_keep_page():
    seg = Segment(text=" ".join(f"w{i}" for i in range(1000)),
                  source_type="pdf", document="a.pdf", page=3)
    chunks = chunk_segments([seg], chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.page == 3 and c.document == "a.pdf" for c in chunks)


def test_video_chunks_keep_timestamps():
    segs = [Segment(text=" ".join(["word"] * 10), source_type="video", document="v.mp4",
                    timestamp_start_ms=i * 5000, timestamp_end_ms=i * 5000 + 4000, confidence=0.9)
            for i in range(20)]
    chunks = chunk_segments(segs, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.timestamp_start_ms < c.timestamp_end_ms for c in chunks)
    assert chunks[0].timestamp_start_ms == 0