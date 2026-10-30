"""Piper voice download (backend/media/tts.py): a dropped connection is retried and resumed."""
import io
import urllib.error

import pytest

from backend.media import tts


class Resp(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status, self.headers = status, {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def voices(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "voices_dir", lambda: tmp_path)
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)
    return tmp_path


def test_urls_follow_the_voice_repository_layout(monkeypatch):
    monkeypatch.setattr(tts, "video_cfg", lambda: {})
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    j, m = tts.voice_urls("tr_TR-dfki-medium")
    assert m.startswith("https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/"
                        "tr_TR-dfki-medium.onnx")
    assert ".onnx.json" in j
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    assert tts.voice_urls("en_US-lessac-medium")[1].startswith("https://mirror.example/rhasspy/")
    with pytest.raises(ValueError):
        tts.voice_urls("not-a-voice")


def test_reset_connection_is_retried_and_resumed(voices, monkeypatch):
    data = b"0123456789" * 100
    seen = []

    def fake_urlopen(req, timeout=0):
        rng = req.get_header("Range")
        seen.append(rng)
        if len(seen) == 1:   # first try: half the file, then the connection is reset
            class Broken(Resp):
                def read(self, n=-1):
                    chunk = super().read(n if n > 0 else 500)
                    if self.tell() >= 500:
                        if getattr(self, "_sent", False):
                            raise ConnectionResetError(10054, "connection forcibly closed")
                        self._sent = True
                    return chunk
            return Broken(data[:600])
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return Resp(data[start:], 206 if rng else 200)

    monkeypatch.setattr(tts.urllib.request, "urlopen", fake_urlopen)
    dest = voices / "x.onnx"
    tts._fetch("https://h/x.onnx", dest)
    assert dest.read_bytes() == data and seen[0] is None and seen[1].startswith("bytes=")
    assert not (voices / "x.onnx.part").exists()


def test_gives_up_with_a_helpful_message(voices, monkeypatch):
    def down(req, timeout=0):
        raise urllib.error.URLError(ConnectionResetError(10054, "reset"))
    monkeypatch.setattr(tts.urllib.request, "urlopen", down)
    with pytest.raises(RuntimeError, match="download_voices.py"):
        tts._fetch("https://h/y.onnx", voices / "y.onnx")


def test_voice_ready(voices, monkeypatch):
    monkeypatch.setattr(tts, "voice_name", lambda lang: "tr_TR-dfki-medium")
    assert not tts.voice_ready("tr")
    (voices / "tr_TR-dfki-medium.onnx").write_bytes(b"1")
    (voices / "tr_TR-dfki-medium.onnx.json").write_bytes(b"{}")
    assert tts.voice_ready("tr")


def test_unknown_voice_fails_at_once(voices, monkeypatch):
    calls = []

    def missing(req, timeout=0):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(tts.urllib.request, "urlopen", missing)
    with pytest.raises(RuntimeError, match="does not exist"):
        tts._fetch("https://h/z.onnx", voices / "z.onnx")
    assert len(calls) == 1


@pytest.mark.parametrize("setting,language,expected", [
    (0.92, "tr", 0.92),                          # one speed for every voice
    ({"tr": 0.9, "en": 0.95}, "tr", 0.9),        # one per language (config.yaml)
    ({"tr": 0.9, "en": 0.95}, "en", 0.95),
    ({"tr": 0.9, "en": 0.95}, "de", 0.95),       # no entry: the English one
    (None, "tr", 0.92), ("fast", "tr", 0.92),    # unusable value: the default
    (9, "en", 2.0),                              # kept within 0.5-2.0
])
def test_voice_speed_per_language(monkeypatch, setting, language, expected):
    monkeypatch.setattr(tts, "video_cfg", lambda: {} if setting is None else {"voice_speed": setting})
    assert tts.voice_speed(language) == expected
