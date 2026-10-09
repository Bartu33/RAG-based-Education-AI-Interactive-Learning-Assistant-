from dataclasses import replace

from backend.config import get_config
from backend.schemas import Segment

TOKENS_PER_WORD = 1.3  # rough estimate; swap for a real tokenizer if needed


def count_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def _words(tokens: int) -> int:
    return max(1, int(tokens / TOKENS_PER_WORD))


def _chunk_text(segments: list[Segment], size: int, overlap: int) -> list[Segment]:
    """Sliding word window inside each page / slide / section (metadata preserved)."""
    out: list[Segment] = []
    step = max(1, size - overlap)
    for seg in segments:
        words = seg.text.split()
        if len(words) <= size:
            out.append(seg)
            continue
        for start in range(0, len(words), step):
            out.append(replace(seg, text=" ".join(words[start:start + size])))
            if start + size >= len(words):
                break
    return out


def _merge(buf: list[Segment]) -> Segment:
    confs = [s.confidence for s in buf if s.confidence is not None]
    return replace(
        buf[0],
        text=" ".join(s.text for s in buf),
        timestamp_start_ms=buf[0].timestamp_start_ms,
        timestamp_end_ms=buf[-1].timestamp_end_ms,
        confidence=round(sum(confs) / len(confs), 3) if confs else None,
    )


def _chunk_video(segments: list[Segment], size: int, overlap: int) -> list[Segment]:
    """Merge consecutive transcript segments into ~size-word chunks keeping start/end times."""
    out: list[Segment] = []
    buf: list[Segment] = []
    words = 0
    for seg in segments:
        n = len(seg.text.split())
        if buf and (words + n > size or seg.document != buf[-1].document):
            out.append(_merge(buf))
            carry: list[Segment] = []
            carried = 0
            if seg.document == buf[-1].document:
                for s in reversed(buf):
                    sw = len(s.text.split())
                    if carried + sw > overlap:
                        break
                    carry.insert(0, s)
                    carried += sw
            buf, words = carry, carried
        buf.append(seg)
        words += n
    if buf:
        out.append(_merge(buf))
    return out


def chunk_segments(segments: list[Segment], chunk_size: int | None = None,
                   overlap: int | None = None) -> list[Segment]:
    cfg = get_config()["indexing"]
    size = _words(chunk_size or cfg["chunk_size"])
    ov = min(_words(overlap if overlap is not None else cfg["chunk_overlap"]), size - 1)
    text_segs = [s for s in segments if s.source_type != "video"]
    video_segs = [s for s in segments if s.source_type == "video"]
    return _chunk_text(text_segs, size, ov) + _chunk_video(video_segs, size, ov)