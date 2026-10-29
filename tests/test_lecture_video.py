"""Explanation video (backend/tasks/lecture_video.py): real ffmpeg + PyMuPDF, fake voice (no model
download), a tiny generated clip standing in for the lecture recording."""
import re
import subprocess

import pytest

fitz = pytest.importorskip("fitz")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("matplotlib")

from backend.media import tts  # noqa: E402
from backend.tasks import lecture_video as lv  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
LINE1 = "Packet switching versus circuit switching"
LINE2 = "Circuit switching reserves resources for a call"


def duration(path) -> float:
    out = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """uploads/CENG336 with a PDF and a 6 s 'lecture recording'; videos go to tmp/generated;
    the voice writes 1.5 s of silence per sentence."""
    import backend.tasks.study as study
    monkeypatch.setattr(study, "resolve_path", lambda p: tmp_path / "uploads")
    monkeypatch.setattr(lv, "resolve_path", lambda p: tmp_path / "generated")
    folder = tmp_path / "uploads" / "CENG336"
    folder.mkdir(parents=True)
    doc = fitz.open()
    pg = doc.new_page()
    pg.insert_text((72, 100), LINE1)
    pg.insert_text((72, 130), LINE2)
    doc.save(folder / "Chapter_1.pdf")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10",
                    "-t", "6", "-pix_fmt", "yuv420p", str(folder / "Lecture01.mp4")], check=True)
    spoken = []

    def fake_say(text, language, out_path, pause_s=0.0):
        spoken.append((language, text))
        return tts.silence(1.5, out_path)
    monkeypatch.setattr(tts, "synthesize", fake_say)
    monkeypatch.setattr(tts, "available", lambda: None)
    monkeypatch.setattr(tts, "voice_name", lambda lang: f"{lang}-test-voice")
    monkeypatch.setattr(lv, "video_cfg", lambda: {**lv.DEFAULTS, "width": 640, "height": 360})
    lv._JOBS.clear()
    return {"tmp": tmp_path, "spoken": spoken}


PAGE_BOARD = {
    "view": "page", "language": "tr",
    "video": {"document": "Lecture01.mp4", "start_ms": 1000, "end_ms": 5000},
    "page": {"document": "Chapter_1.pdf", "type": "pdf", "page": 1, "renderable": True,
             "highlights": [LINE1, LINE2]},
    "lesson": {"beats": [{"line": None, "say": "Bunu videoda anlatıyorum."},
                         {"line": LINE1, "say": "Şu satıra bakın."},
                         {"line": LINE2, "say": "Devre anahtarlama kaynak ayırır."}],
               "closing": "Fark bu.", "narrated": True}}

WB_BOARD = {
    "view": "whiteboard", "language": "en", "video": None, "page": None,
    "whiteboard": {"title": "Transmission delay",
                   "steps": [{"write": "$d = \\dfrac{L}{R}$", "say": "First the formula."},
                             {"write": "$d = 1000/100 = 10$ s", "say": ""}],
                   "result": "$d = 10$ s", "closing": "So it takes ten seconds."}}


def test_script_from_page_lesson():
    scenes = lv.build_script(PAGE_BOARD, "q")
    assert scenes[0]["layout"] == "intro" and scenes[0]["say"] == lv.TEXT["tr"]["intro"]
    scenes = scenes[1:]
    assert [s["say"] for s in scenes] == ["Bunu videoda anlatıyorum.", "Şu satıra bakın.",
                                          "Devre anahtarlama kaynak ayırır.", "Fark bu."]
    assert scenes[1]["highlights"] == [LINE1] and scenes[0]["highlights"] == []
    assert scenes[-1]["final"] and scenes[-1]["highlights"] == [LINE1, LINE2]
    assert [s["step"] for s in scenes] == [1, 2, 3, 4] and scenes[0]["steps"] == 4


def test_script_from_whiteboard_speaks_formula_when_no_sentence():
    scenes = lv.build_script(WB_BOARD, "q")[1:]
    assert [s["steps_shown"] for s in scenes] == [1, 2, 2]
    assert scenes[1]["say"] == "d equals 1000 over 100 equals 10 s"   # formula read out in words
    assert scenes[-1]["final"] and scenes[-1]["say"] == "So it takes ten seconds."


def test_long_sentences_become_subtitle_pieces():
    text = ("First, digital circuits are less sensitive to changes in component values, and they are "
            "also less sensitive to temperature, ageing and other external parameters. That is the point.")
    pieces = lv.split_subtitles(text, 90)
    assert len(pieces) >= 2 and all(len(p) <= 90 for p in pieces)
    assert " ".join(pieces).split() == text.split()            # nothing lost
    assert lv.split_subtitles("Short one.", 90) == ["Short one."]


def test_no_explanation_no_video():
    assert lv.build_script({"view": "none"}, "q") == []
    with pytest.raises(ValueError):
        lv.start("q", "CENG336", {"view": "none"})


def test_frames_have_video_size(env):
    from backend.tasks.board import render_page
    png = lambda lines, pi=0: render_page("CENG336", "Chapter_1.pdf", 1, lines)  # noqa: E731
    wb = {**WB_BOARD, "whiteboard": {**WB_BOARD["whiteboard"], "given": ["$L = 1000$ bit", "R=100\\,bps"],
                                     "find": "$d$", "check": "Units: bits / (bits/s) = s."}}
    for board in (PAGE_BOARD, wb):
        for sc in lv.build_script(board, "Bir soru?"):
            if sc["layout"] == "page" and sc["highlights"]:
                sc["box"], sc["content"] = (0.1, 0.1, 0.6, 0.15), (0.1, 0.1, 0.8, 0.3)
            im = lv.render_frame(sc, board, "CENG336", "Bir soru?", board is PAGE_BOARD, png,
                                 {**lv.DEFAULTS, "width": 1280, "height": 720})
            assert im.size == (1280, 720)
    im = lv.render_frame(lv.build_script(wb, "q")[1], wb, "C", "q", False, png, dict(lv.DEFAULTS))
    assert im.size == (1920, 1080)                                 # Full HD by default


def test_formulas_without_dollars_are_drawn_as_math():
    from backend.media.mathfmt import fix_line, for_mathtext
    raw = "x1(n)\\,x2(n)=\\{1\\times2,\\,-3\\times(-1)\\}"
    assert fix_line(raw) == "$" + raw.replace("\\,", " ") + "$"   # LaTeX spacing becomes a plain space
    assert fix_line("Toplam: \\frac{a}{b} olur") == "Toplam: $\\frac{a}{b}$ olur"
    assert fix_line("\\(|z| = 5\\)") == "$|z| = 5$" and fix_line("x(n) = {1, 2}") == "x(n) = {1, 2}"
    assert for_mathtext("$\\boxed{y = 5\\text{ m}}$") == "$y = 5\\mathrm{\\ m}$"
    plain = lv._formula_image("plain words", 30)
    math = lv._formula_image(raw, 30)
    assert math.width < lv._formula_image(raw.replace("\\", "\\\\"), 30).width or math.height > plain.height


def test_latex_spacing_never_shows_as_a_comma():
    from backend.media.mathfmt import fix_line, for_mathtext
    line = "$a\\, y_1[n] + b\\;y_2[n] = a\\,x_1[n]$"
    assert fix_line(line) == "$a  y_1[n] + b y_2[n] = a x_1[n]$"
    assert "\\," not in for_mathtext(line)
    assert fix_line("a\\,y_1[n] + b\\,y_2[n]") == "$a y_1[n] + b y_2[n]$"   # bare LaTeX too
    assert fix_line("$a, b \\in R$") == "$a, b \\in R$"                    # a real comma stays
    assert fix_line("$\\{1, 2\\}$") == "$\\{1, 2\\}$"                   # braces are not spacing


def test_page_video_with_lecture_recording_in_corner(env):
    out = env["tmp"] / "v.mp4"
    res = lv.make_video("Paket anahtarlama nedir?", "CENG336", PAGE_BOARD, out)
    assert out.is_file() and res["pip"] and res["scenes"] == 5 and res["voice"] == "tr-test-voice"
    assert abs(duration(out) - 5 * 1.5) < 0.6                  # each piece as long as its sentence
    assert [lang for lang, _ in env["spoken"]] == ["tr"] * 5
    info = subprocess.run([FF, "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
    assert "h264" in info and "aac" in info and "640x360" in info


def test_whiteboard_video_and_missing_recording(env):
    (env["tmp"] / "uploads" / "CENG336" / "Lecture01.mp4").unlink()
    board = {**WB_BOARD, "video": {"document": "Lecture01.mp4", "start_ms": 0, "end_ms": 3000}}
    res = lv.make_video("Calculate the delay", "CENG336", board, env["tmp"] / "w.mp4")
    assert not res["pip"] and any("not on the server" in n for n in res["notes"])
    assert abs(duration(env["tmp"] / "w.mp4") - 4 * 1.5) < 0.6


def test_no_voice_still_makes_a_subtitled_video(env, monkeypatch):
    monkeypatch.setattr(tts, "available", lambda: "piper-tts is not installed")
    res = lv.make_video("q", "CENG336", WB_BOARD, env["tmp"] / "s.mp4")
    assert res["voice"] is None and any("subtitles only" in n for n in res["notes"])
    assert duration(env["tmp"] / "s.mp4") > 3


def test_background_job_api_and_cache(env):
    from fastapi.testclient import TestClient
    import backend.main as main
    client = TestClient(main.app)
    r = client.post("/lecture-video", json={"question": "Paket?", "course": "CENG336", "board": PAGE_BOARD})
    assert r.status_code == 200 and r.json()["state"] in ("queued", "running", "done")
    jid = r.json()["id"]
    assert lv.wait(jid, 60)["state"] == "done"
    st = client.get(f"/lecture-video/{jid}").json()
    assert st["state"] == "done" and st["pip"] and st["scenes"] == 5
    f = client.get(f"/lecture-video/{jid}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "video/mp4" and len(f.content) > 1000

    lv._JOBS.clear()   # backend restarted: the cached file is reused, nothing is rebuilt
    again = client.post("/lecture-video", json={"question": "Paket?", "course": "CENG336", "board": PAGE_BOARD})
    assert again.json()["state"] == "done" and again.json()["id"] == jid

    assert client.get("/lecture-video/nope").status_code == 404
    assert client.get("/lecture-video/nope/file").status_code == 404
    bad = client.post("/lecture-video", json={"question": "q", "course": "CENG336", "board": {"view": "none"}})
    assert bad.status_code == 400


def test_voiceless_video_is_built_again_next_time(env, monkeypatch):
    calls = {"n": 0}
    real = lv.make_video

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(lv, "make_video", counting)
    monkeypatch.setattr(tts, "available", lambda: "voice unavailable (download reset)")
    job = lv.start("Paket?", "CENG336", PAGE_BOARD)
    done = lv.wait(job["id"], 60)
    assert done["state"] == "done" and done["voice"] is None
    assert not lv.output_path("CENG336", job["id"]).is_file()            # not kept as the final video
    assert lv.file_for(job["id"]).name.endswith(".silent.mp4")
    monkeypatch.setattr(tts, "available", lambda: None)                  # the voice works now
    again = lv.wait(lv.start("Paket?", "CENG336", PAGE_BOARD)["id"], 60)
    assert calls["n"] == 2 and again["voice"] == "tr-test-voice"
    assert lv.output_path("CENG336", job["id"]).is_file()


def test_voice_follows_the_language_of_the_explanation(env):
    english = {**PAGE_BOARD, "lesson": {"beats": [{"line": LINE1, "say": "Look at this line, it is the key."}],
                                         "closing": "That is the difference.", "narrated": True}}
    res = lv.make_video("Paket anahtarlama nedir?", "CENG336", english, env["tmp"] / "e.mp4")
    assert res["voice"] == "en-test-voice" and {lang for lang, _ in env["spoken"]} == {"en"}


def test_text_language():
    from backend.tasks.board import text_language
    assert text_language("Şimdi bu satıra bakalım, devre anahtarlama kaynak ayırır.") == "tr"
    assert text_language("Now let us look at this line, which is the key idea.") == "en"


def test_formulas_in_sentences_are_never_read_as_dollars(env, monkeypatch):
    from backend.media.mathfmt import plain_sentence, spoken_sentence
    say = "Aynı konumdaki $x_2$ örneği iki, $\\frac{L}{R}$ ise gecikme."
    assert plain_sentence(say) == "Aynı konumdaki x₂ örneği iki, L/R ise gecikme."
    assert spoken_sentence(say, "tr") == "Aynı konumdaki x 2 örneği iki, L bölü R ise gecikme."
    subs = []
    real = lv.render_frame
    monkeypatch.setattr(lv, "render_frame", lambda *a, **k: subs.append(k.get("subtitle")) or real(*a, **k))
    board = {**PAGE_BOARD, "lesson": {"beats": [{"line": LINE1, "say": say}], "closing": "", "narrated": True}}
    lv.make_video("Paket?", "CENG336", board, env["tmp"] / "m.mp4")
    assert all("$" not in t and "\\" not in t for _, t in env["spoken"])
    assert all("$" not in s and "\\" not in s for s in subs) and "x₂" in " ".join(subs)
