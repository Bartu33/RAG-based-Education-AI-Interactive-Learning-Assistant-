"""Lecturer voice: text -> WAV with Piper (local, CPU, no data leaves the machine).

The voice model (.onnx + .onnx.json, ~60 MB) is downloaded once from the Piper voice repository on
first use into `video.voices_dir` (default data/models/piper). Voices per language come from
config.yaml -> video.voices. The download retries and resumes (a dropped connection, e.g.
"WinError 10054", only costs a retry); `python scripts/download_voices.py` fetches them up front.
A different mirror can be set with video.voice_base_url or the HF_ENDPOINT environment variable.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.request
import wave
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from backend.config import get_config, resolve_path

log = logging.getLogger(__name__)
_LOCK = threading.Lock()          # one synthesis at a time (Piper's session is not re-entrant)
_DL_LOCK = threading.Lock()       # one download at a time

DEFAULT_VOICES = {"tr": "tr_TR-dfki-medium", "en": "en_US-lessac-medium"}   # the only Turkish Piper voice
_VOICE = re.compile(r"^(?P<family>[^-_]+)_(?P<region>[^-]+)-(?P<name>[^-]+)-(?P<quality>.+)$")
RETRIES = 5


def video_cfg() -> dict:
    return get_config().get("video") or {}


def voice_name(language: str) -> str:
    voices = {**DEFAULT_VOICES, **(video_cfg().get("voices") or {})}
    return voices.get(language) or voices["en"]


def voices_dir() -> Path:
    return resolve_path(video_cfg().get("voices_dir", "data/models/piper"))


def voice_files(name: str) -> tuple[Path, Path]:
    folder = voices_dir()
    return folder / f"{name}.onnx", folder / f"{name}.onnx.json"


def voice_ready(language: str) -> bool:
    """The voice for this language is on disk (no download needed)."""
    return all(p.is_file() and p.stat().st_size > 0 for p in voice_files(voice_name(language)))


def voice_urls(name: str) -> list[str]:
    m = _VOICE.match(name)
    if not m:
        raise ValueError(f"Not a Piper voice name: {name}")
    base = (video_cfg().get("voice_base_url") or
            (os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/rhasspy/piper-voices/resolve/main"))
    lang = f"{m['family']}_{m['region']}"
    folder = f"{base.rstrip('/')}/{m['family']}/{lang}/{m['name']}/{m['quality']}/{name}"
    return [folder + ".onnx.json?download=true", folder + ".onnx?download=true"]


def _fetch(url: str, dest: Path, progress: Optional[Callable[[str, int, int], None]] = None) -> None:
    """Download url -> dest, resuming a partial file and retrying a dropped connection."""
    part = dest.with_name(dest.name + ".part")
    last: Exception = RuntimeError("no attempt")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "rag-learning-assistant",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:   # server ignored the range: start over
                    have = 0
                total = have + int(resp.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    while chunk := resp.read(1 << 16):
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(dest.name, done, total)
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001  (reset / timeout / HTTP error)
            last = exc
            if getattr(exc, "code", None) == 416:   # the partial file is already complete
                part.replace(dest)
                return
            if getattr(exc, "code", None) == 404:   # wrong voice name: retrying will not help
                raise RuntimeError(f"{dest.name} does not exist in the Piper voice repository (404); "
                                   "check video.voices in config.yaml (Turkish: tr_TR-dfki-medium)") from exc
            log.warning("Voice download %s failed (attempt %d/%d): %s", dest.name, attempt, RETRIES, exc)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"could not download {dest.name} after {RETRIES} attempts ({last}). "
                       f"Run 'python scripts/download_voices.py' or put the files in {dest.parent}")


def download(name: str, progress: Optional[Callable[[str, int, int], None]] = None) -> tuple[Path, Path]:
    """Make sure the voice files are on disk; returns (model, config)."""
    model, config = voice_files(name)
    with _DL_LOCK:
        model.parent.mkdir(parents=True, exist_ok=True)
        for url, dest in zip(voice_urls(name), (config, model)):
            if not (dest.is_file() and dest.stat().st_size > 0):
                log.info("Downloading Piper voice file %s", dest.name)
                _fetch(url, dest, progress)
    return model, config


@lru_cache(maxsize=4)
def _voice(name: str):
    from piper import PiperVoice   # piper-tts
    model, _ = download(name)
    return PiperVoice.load(str(model))


def voice_speed(language: str) -> float:
    """video.voice_speed: one number for every voice, or one per language ({tr: 0.9, en: 0.95}).
    1.0 = Piper default, 0.9 = a bit slower / clearer."""
    speed = video_cfg().get("voice_speed", 0.92)
    if isinstance(speed, dict):
        speed = speed.get(language, speed.get("en", 0.92))
    try:
        return max(0.5, min(float(speed), 2.0))
    except (TypeError, ValueError):
        return 0.92


def _syn_config(language: str = "en"):
    speed = voice_speed(language)
    try:
        from piper import SynthesisConfig
        return SynthesisConfig(length_scale=1.0 / speed)
    except Exception:  # noqa: BLE001  (older piper-tts)
        return None


def synthesize(text: str, language: str, out_path: Path, pause_s: float = 0.0) -> float:
    """Write `text` spoken by the lecturer voice to `out_path` (WAV). Returns the duration in s.
    `pause_s` of silence is added at the end (a breath between two steps)."""
    voice = _voice(voice_name(language))
    out_path = Path(out_path)
    cfg = _syn_config(language)
    with _LOCK, wave.open(str(out_path), "wb") as wf:
        if cfg is not None:
            voice.synthesize_wav(text, wf, syn_config=cfg)
        else:
            voice.synthesize_wav(text, wf)
        rate, width, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        if pause_s > 0:
            wf.writeframes(b"\x00" * int(rate * pause_s) * width * channels)
    return wav_seconds(out_path)


def silence(seconds: float, out_path: Path, rate: int = 22050) -> float:
    """A silent mono WAV (used when no voice is available: the video then relies on subtitles)."""
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * seconds))
    return wav_seconds(out_path)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def available() -> Optional[str]:
    """None when Piper can be used, else the reason (shown in the UI)."""
    try:
        import piper  # noqa: F401
    except ImportError:
        return "piper-tts is not installed (pip install piper-tts)"
    return None
