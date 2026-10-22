"""A student's lecture video (backend/tasks/video_upload.py, /videos endpoints) and video questions
in the study buddy. Whisper, the embedding model and the store are fakes."""
import numpy as np
import pytest

from backend.indexing.chunker import RegexTokenizer, chunk_segments
from backend.indexing.vectorstore import chunk_id
from backend.schemas import Segment
from backend.tasks import study, video_upload
from tests.test_study import FakeStore, _meta


class Store:
    """Keeps upserted chunks by id, like Chroma."""

    def __init__(self):
        self.rows, self.deleted = {}, []

    def add(self, chunks, embeddings, ids=None):
        for i, c, e in zip(ids, chunks, embeddings):
            self.rows[i] = c.text
        return len(chunks)

    def delete_document(self, document, course=None):
        self.deleted.append((course, document))


def _chunk(segments):
    return chunk_segments(segments, chunk_size=30, overlap=6, tokenizer=RegexTokenizer())


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(video_upload, "chunk_segments", _chunk)
    monkeypatch.setattr(video_upload, "embed_texts", lambda texts: np.zeros((len(texts), 3)))
    monkeypatch.setattr(video_upload, "upload_cfg", lambda: {"beam_size": 1, "step_s": 30})


def _segments(n=12):
    return [Segment(f"part {i} of the lecture about sampling and filters", "video", "L1.mp4",
                    timestamp_start_ms=i * 10_000, timestamp_end_ms=(i + 1) * 10_000) for i in range(n)]


def test_transcript_is_indexed_in_steps_and_ends_like_a_one_shot_index():
    store, seen = Store(), []

    def transcribe(path, beam_size):
        def gen():
            for s in _segments():
                seen.append(len(store.rows))      # how much the buddy could see at this moment
                yield s
        return 120_000, gen()
    job = {"course": "CENG336", "lecture": 2, "document": "L1.mp4", "state": "queued", "done_ms": 0,
           "duration_ms": None, "chunks": 0, "error": None, "finished": None}
    video_upload.run(job, "L1.mp4", store=store, transcribe=transcribe)

    assert job["state"] == "done" and job["done_ms"] == 120_000 and store.deleted == [("CENG336", "L1.mp4")]
    assert 0 < seen[5] < seen[-1]                 # the first minutes were searchable before the end
    expected = _chunk([Segment(s.text, "video", "L1.mp4", timestamp_start_ms=s.timestamp_start_ms,
                               timestamp_end_ms=s.timestamp_end_ms, course="CENG336", lecture=2)
                       for s in _segments()])
    assert job["chunks"] == len(expected)
    assert store.rows == {chunk_id("CENG336", 2, "L1.mp4", i): c.text for i, c in enumerate(expected)}


def test_silent_video_fails_with_a_message():
    job = {"course": "C", "lecture": 1, "document": "x.mp4", "state": "queued", "done_ms": 0,
           "duration_ms": None, "chunks": 0, "error": None, "finished": None}
    video_upload.run(job, "x.mp4", store=Store(), transcribe=lambda p, b: (5000, iter([])))
    assert job["state"] == "failed" and "No speech" in job["error"]


def test_upload_endpoint(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.main as main
    started = []
    monkeypatch.setattr(main, "resolve_path", lambda p: tmp_path)
    monkeypatch.setattr(video_upload, "start", lambda path, course, lecture: started.append(
        (path.name, course, lecture)) or {"id": "j1", "state": "queued"})
    client = TestClient(main.app)
    r = client.post("/videos", data={"course": "CENG336", "lecture": "3"},
                    files={"file": ("lec 3.mp4", b"\x00\x01", "video/mp4")})
    assert r.status_code == 200 and started == [("lec 3.mp4", "CENG336", 3)]
    assert (tmp_path / "CENG336" / "lec 3.mp4").read_bytes() == b"\x00\x01"
    bad = client.post("/videos", data={"course": "CENG336", "lecture": "3"},
                      files={"file": ("notes.pdf", b"%PDF", "application/pdf")})
    assert bad.status_code == 400
    assert client.get("/videos/jobs/nope").status_code == 404
    assert client.get("/videos/status", params={"course": "X", "document": "y.mp4"}).json() == {"job": None}


VIDEO = [("Intro: what is a signal.", _meta("L1.mp4", kind="video", start=0, end=60_000), [1, 0, 0]),
         ("Sampling theorem and aliasing.", _meta("L1.mp4", kind="video", start=60_000, end=120_000), [0, 1, 0]),
         ("Low-pass filters before sampling.", _meta("L1.mp4", kind="video", start=120_000, end=180_000), [0, 0.8, 0.6]),
         ("Quantisation noise.", _meta("L1.mp4", kind="video", start=600_000, end=660_000), [0, 0.95, 0.31]),
         ("Wrap-up and homework.", _meta("L1.mp4", kind="video", start=900_000, end=960_000), [0, 0, 1])]


@pytest.fixture()
def video_store(monkeypatch):
    study._CACHE.clear()
    import backend.indexing.embedder as emb
    monkeypatch.setattr(emb, "embed_query", lambda text: [0.0, 1.0, 0.0])   # "about sampling"
    monkeypatch.setattr(study, "study_cfg", lambda: {**study.DEFAULTS, "top_context": 2})
    return FakeStore(VIDEO)


def test_video_question_prefers_the_part_just_watched_and_includes_the_moment(video_store):
    plain = study.select_chunks("CENG336", "L1.mp4", "ask", "aliasing?", store=video_store)["chunks"]
    assert [c["text"] for c in plain] == ["Sampling theorem and aliasing.", "Quantisation noise."]
    at = study.select_chunks("CENG336", "L1.mp4", "ask", "what does he mean here?", time_ms=930_000,
                             store=video_store)
    texts = [c["text"] for c in at["chunks"]]
    assert texts[0] == "Sampling theorem and aliasing." and "Wrap-up and homework." in texts   # the moment
    watched = study.select_chunks("CENG336", "L1.mp4", "ask", "q", time_ms=660_000, store=video_store)
    assert watched["chunks"][0]["text"] == "Quantisation noise."         # just watched beats a bit better match


def test_video_question_prompt_names_the_moment(video_store):
    prep = study.prepare("CENG336", "L1.mp4", "ask", "what is this?", time_ms=125_000, store=video_store)
    assert "now at minute 02:05 of the lecture video" in prep["messages"][-1]["content"]
    doc = study.prepare("CENG336", "L1.mp4", "ask", "what is this?", store=video_store)
    assert "now at" not in doc["messages"][-1]["content"]


def test_old_faster_whisper_with_new_pyav_falls_back_to_ffmpeg(monkeypatch):
    """faster-whisper < 1.1.1 + PyAV 14: "open() got an unexpected keyword argument 'metadata_errors'"."""
    from backend.ingestion import video_stt
    seen = []

    class Model:
        def transcribe(self, audio, **kw):
            seen.append(audio)
            if isinstance(audio, str):
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
            return iter([]), type("Info", (), {"duration": 2.0})

    monkeypatch.setattr(video_stt, "_get_model", lambda: Model())
    monkeypatch.setattr(video_stt, "decode_audio", lambda path: np.zeros(32000, dtype=np.float32))
    assert video_stt.iter_transcript("L1.mp4")[0] == 2000
    assert isinstance(seen[0], str) and isinstance(seen[1], np.ndarray)
