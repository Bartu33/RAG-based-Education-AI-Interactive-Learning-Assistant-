from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """One piece of extracted text, later split into chunks. Same class is used for chunks."""
    text: str
    source_type: str                      # pdf | pptx | docx | video
    document: str
    page: Optional[int] = None            # page (pdf) or slide number (pptx)
    section: Optional[str] = None         # heading / slide title
    timestamp_start_ms: Optional[int] = None
    timestamp_end_ms: Optional[int] = None
    confidence: Optional[float] = None    # video only, 0..1
    course: str = ""
    lecture: int = 0

    def to_metadata(self) -> dict:
        """Chroma metadata cannot contain None, so missing values become -1 / ''."""
        return {
            "course": self.course,
            "lecture": int(self.lecture),
            "source_type": self.source_type,
            "document": self.document,
            "section": self.section or "",
            "page": -1 if self.page is None else int(self.page),
            "timestamp_start_ms": -1 if self.timestamp_start_ms is None else int(self.timestamp_start_ms),
            "timestamp_end_ms": -1 if self.timestamp_end_ms is None else int(self.timestamp_end_ms),
            "confidence": -1.0 if self.confidence is None else float(self.confidence),
        }


def metadata_to_public(meta: dict) -> dict:
    out = dict(meta)
    for key in ("page", "timestamp_start_ms", "timestamp_end_ms", "confidence"):
        if out.get(key, -1) in (-1, -1.0):
            out[key] = None
    out["section"] = out.get("section") or None
    return out


def format_ms(ms: int) -> str:
    s = int(ms // 1000)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def format_location(meta: dict) -> str:
    """Human-readable source label used in prompts and in the UI."""
    kind, doc = meta["source_type"], meta["document"]
    if kind == "video" and meta.get("timestamp_start_ms") is not None:
        return f"{doc} @ {format_ms(meta['timestamp_start_ms'])}-{format_ms(meta['timestamp_end_ms'])}"
    if kind == "pdf" and meta.get("page"):
        return f"{doc}, p. {meta['page']}"
    if kind == "pptx" and meta.get("page"):
        return f"{doc}, slide {meta['page']}"
    if meta.get("section"):
        return f"{doc}, {meta['section']}"
    return doc


def chunk_to_source(hit: dict, max_chars: int = 400) -> dict:
    """Serialize a retrieved chunk for the API / UI."""
    m = hit["metadata"]
    return {
        "type": m["source_type"],
        "document": m["document"],
        "page": m.get("page"),
        "start_ms": m.get("timestamp_start_ms"),
        "end_ms": m.get("timestamp_end_ms"),
        "score": round(float(hit["score"]), 3),
        "text": hit["text"][:max_chars],
    }