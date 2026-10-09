import math
from functools import lru_cache
from pathlib import Path

from backend.config import get_config
from backend.schemas import Segment


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel  # heavy import, load lazily
    cfg = get_config()["ingestion"]
    return WhisperModel(cfg["whisper_model"],
                        device=cfg["whisper_device"],
                        compute_type=cfg["whisper_compute_type"])


def transcribe_video(path) -> list[Segment]:
    """Time-coded transcript segments. confidence = exp(avg_logprob) in 0..1."""
    path = Path(path)
    cfg = get_config()["ingestion"]
    model = _get_model()
    raw_segments, _info = model.transcribe(str(path), language=cfg["language"],
                                           vad_filter=True, beam_size=5)
    segments: list[Segment] = []
    for seg in raw_segments:  # generator: transcription happens while iterating
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(
            text=text, source_type="video", document=path.name,
            timestamp_start_ms=int(seg.start * 1000),
            timestamp_end_ms=int(seg.end * 1000),
            confidence=round(math.exp(seg.avg_logprob), 3),
        ))
    return segments