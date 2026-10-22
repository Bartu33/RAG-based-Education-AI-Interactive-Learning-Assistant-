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


def decode_audio(path, sampling_rate: int = 16000):
    """The audio track as 16 kHz mono float32 samples, decoded by the ffmpeg program."""
    import shutil
    import subprocess

    import numpy as np
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg), or upgrade faster-whisper")
    out = subprocess.run([exe, "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sampling_rate),
                          "-f", "f32le", "-loglevel", "error", "-"], capture_output=True, check=False)
    if out.returncode != 0:
        raise ValueError(f"Could not read the audio: {out.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(out.stdout, dtype=np.float32)


def iter_transcript(path, beam_size: int | None = None):
    """(duration in ms, generator of time-coded Segments). The transcription runs while the
    generator is consumed, so a caller can index the first minutes before the rest is done.
    confidence = exp(avg_logprob) in 0..1."""
    path = Path(path)
    cfg = get_config()["ingestion"]
    options = {"language": cfg["language"], "vad_filter": True,
               "beam_size": int(beam_size or cfg.get("whisper_beam_size", 5))}
    try:
        raw_segments, info = _get_model().transcribe(str(path), **options)
    except TypeError as exc:
        # faster-whisper < 1.1.1 passes `metadata_errors` to PyAV, which PyAV 14+ no longer accepts
        if "metadata_errors" not in str(exc):
            raise
        raw_segments, info = _get_model().transcribe(decode_audio(path), **options)

    def segments():
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                yield Segment(text=text, source_type="video", document=path.name,
                              timestamp_start_ms=int(seg.start * 1000),
                              timestamp_end_ms=int(seg.end * 1000),
                              confidence=round(math.exp(seg.avg_logprob), 3))
    return int(info.duration * 1000), segments()


def transcribe_video(path) -> list[Segment]:
    """Time-coded transcript segments of the whole video."""
    return list(iter_transcript(path)[1])
