"""Token-based chunking.

Chunk sizes are measured with the tokenizer of the embedding model (BGE-m3 / XLM-R
SentencePiece), so `chunk_size: 500` in config.yaml means 500 real model tokens.
If the tokenizer cannot be loaded (offline machine, no HF cache), a regex word/punctuation
tokenizer is used as an approximate fallback and a warning is logged once.

Text sources (pdf/pptx/docx): sliding token window inside each page / slide / section, cut at a
sentence boundary when one is close to the window end.
Video: consecutive transcript segments are merged until the token budget is reached, keeping the
first start and last end timestamp; overlap is carried as whole transcript segments.
"""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from functools import lru_cache
from typing import Optional, Protocol

from backend.config import get_config
from backend.schemas import Segment

log = logging.getLogger(__name__)

_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?;:…]\s*$|\n\s*$")


class Tokenizer(Protocol):
    name: str

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span (start, end) of every token, special tokens excluded."""
        ...


class HFTokenizer:
    """Fast Hugging Face tokenizer (same vocabulary the embedding model uses)."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer  # installed with sentence-transformers
        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self._tok, "is_fast", False):
            raise RuntimeError(f"{model_name} has no fast tokenizer (offsets unavailable)")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=False, return_token_type_ids=False, verbose=False)
        # SentencePiece may emit zero-width pieces (e.g. a lone "▁"); drop them
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]


class RegexTokenizer:
    """Approximate fallback: words and punctuation marks count as one token each."""
    name = "regex-fallback"

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _FALLBACK_RE.finditer(text)]


@lru_cache(maxsize=4)
def get_tokenizer(model_name: Optional[str] = None) -> Tokenizer:
    cfg = get_config()["indexing"]
    name = model_name or cfg.get("tokenizer") or cfg["embedding_model"]
    try:
        return HFTokenizer(name)
    except Exception as exc:  # noqa: BLE001
        log.warning("Tokenizer %s could not be loaded (%s); using approximate regex tokenizer.",
                    name, exc)
        return RegexTokenizer()


def count_tokens(text: str, tokenizer: Optional[Tokenizer] = None) -> int:
    return len((tokenizer or get_tokenizer()).offsets(text))


# --------------------------------------------------------------------------- text sources
def _cut_point(text: str, offs: list[tuple[int, int]], start: int, end: int,
               min_keep: int) -> int:
    """Index (exclusive) where the window [start, end) should end. Searches backwards for a
    token that closes a sentence, but never makes the chunk shorter than `min_keep` tokens."""
    if end >= len(offs):
        return len(offs)
    for i in range(end - 1, start + min_keep - 1, -1):
        nxt = offs[i + 1][0] if i + 1 < len(offs) else len(text)
        if _SENTENCE_END.search(text[offs[i][0]:nxt]):
            return i + 1
    return end


def _chunk_text(segments: list[Segment], size: int, overlap: int, tok: Tokenizer,
                sentence_aware: bool) -> list[Segment]:
    out: list[Segment] = []
    min_keep = max(1, int(size * 0.75))  # a sentence cut may shorten a chunk by at most 25 %
    for seg in segments:
        offs = tok.offsets(seg.text)
        if len(offs) <= size:
            out.append(seg)
            continue
        start = 0
        while start < len(offs):
            end = min(start + size, len(offs))
            if sentence_aware:
                end = _cut_point(seg.text, offs, start, end, min_keep)
            text = seg.text[offs[start][0]:offs[end - 1][1]].strip()
            if text:
                out.append(replace(seg, text=text))
            if end >= len(offs):
                break
            start = max(start + 1, end - overlap)
    return out


# --------------------------------------------------------------------------- video
def _merge(buf: list[Segment]) -> Segment:
    confs = [s.confidence for s in buf if s.confidence is not None]
    return replace(
        buf[0],
        text=" ".join(s.text for s in buf),
        timestamp_start_ms=buf[0].timestamp_start_ms,
        timestamp_end_ms=buf[-1].timestamp_end_ms,
        confidence=round(sum(confs) / len(confs), 3) if confs else None,
    )


def _chunk_video(segments: list[Segment], size: int, overlap: int, tok: Tokenizer) -> list[Segment]:
    out: list[Segment] = []
    buf: list[Segment] = []
    lens: list[int] = []
    total = 0
    for seg in segments:
        n = len(tok.offsets(seg.text))
        if buf and (total + n > size or seg.document != buf[-1].document):
            out.append(_merge(buf))
            carry, carry_lens, carried = [], [], 0
            if seg.document == buf[-1].document:
                for s, sl in zip(reversed(buf), reversed(lens)):
                    if carried + sl > overlap:
                        break
                    carry.insert(0, s)
                    carry_lens.insert(0, sl)
                    carried += sl
            buf, lens, total = carry, carry_lens, carried
        buf.append(seg)
        lens.append(n)
        total += n
    if buf:
        out.append(_merge(buf))
    return out


# --------------------------------------------------------------------------- public API
def chunk_segments(segments: list[Segment], chunk_size: int | None = None,
                   overlap: int | None = None, tokenizer: Optional[Tokenizer] = None,
                   sentence_aware: Optional[bool] = None) -> list[Segment]:
    cfg = get_config()["indexing"]
    tok = tokenizer or get_tokenizer()
    size = max(1, int(chunk_size or cfg["chunk_size"]))
    ov = overlap if overlap is not None else cfg["chunk_overlap"]
    ov = max(0, min(int(ov), size - 1))
    if sentence_aware is None:
        sentence_aware = cfg.get("sentence_aware", True)
    text_segs = [s for s in segments if s.source_type != "video"]
    video_segs = [s for s in segments if s.source_type == "video"]
    return (_chunk_text(text_segs, size, ov, tok, sentence_aware)
            + _chunk_video(video_segs, size, ov, tok))
