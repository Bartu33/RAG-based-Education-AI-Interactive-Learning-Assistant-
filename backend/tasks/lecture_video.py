"""Explanation video: the teaching board turned into an MP4 the system "records" itself.

Script  = the board's lesson (backend/tasks/board.py), no extra LLM call: a short title card, one
          scene per beat (what the lecturer says + the line pointed at on the page, or the line
          written on the whiteboard) and a closing scene with the summary / result.
Voice   = Piper, local (backend/media/tts.py). Long sentences are cut into subtitle-sized pieces,
          each spoken and shown on its own, so nothing is cut off on screen.
Picture = one frame per piece (Pillow): left the slide / page (zoomed to the line being explained,
          which is framed and pointed at) or the whiteboard (formulas drawn with matplotlib); right a
          panel with the key points so far (page) or what is given / asked (whiteboard); the
          sentence as a subtitle; the step counter in the header.
Video   = ffmpeg (bundled by imageio-ffmpeg): each frame stays as long as its sentence is spoken,
          the pieces are joined, the matching part of the original lecture recording plays in the
          bottom-right corner, and the whole video fades in and out.
Videos are built in a background thread and cached under data/generated/<course>/. A video made
without a voice (download failed) is not kept as the final one: asking again tries the voice again.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from backend.config import get_config, resolve_path
from backend.media.mathfmt import for_mathtext, mask_math, plain_sentence, pretty, speakable, unmask

log = logging.getLogger(__name__)
VERSION = 3   # bump when the look changes, so cached videos are rebuilt

DEFAULTS = {
    "enabled": True,
    "width": 1920,
    "height": 1080,
    "fps": 12,
    "pause_s": 0.45,              # silence after each scene
    "chunk_pause_s": 0.2,         # silence between the pieces of one scene
    "seconds_per_word": 0.38,     # piece length when no voice is available
    "subtitle_chars": 90,         # a subtitle piece is at most about this long (two lines)
    "intro": True,                # title card with the question
    "pip": True,                  # original lecture recording in the corner
    "pip_width": 384,             # in 1280-wide units (scaled with the video)
    "max_scenes": 14,
    "output_dir": "data/generated",
}

BG, PANEL, PANEL2, INK, MUTED, ACCENT = ((11, 15, 20), (18, 25, 35), (24, 33, 46), (230, 237, 243),
                                         (139, 152, 169), (45, 212, 191))
BOARD_BG, BOARD_INK, BOARD_TITLE, MARK = (251, 251, 248), (29, 35, 48), (15, 92, 85), (230, 84, 64)

TEXT = {
    "tr": {"label": "Ders anlatımı", "question": "Soru", "points": "Anahtar noktalar", "summary": "Özet",
           "given": "Verilenler", "find": "İstenen", "check": "Kontrol", "result": "Sonuç",
           "rec": "Ders kaydı", "step": "Adım", "steps": "adım", "page": "sayfa", "slide": "slayt",
           "intro": "Bu soruyu birlikte adım adım inceleyelim."},
    "en": {"label": "Lecture", "question": "Question", "points": "Key points", "summary": "Summary",
           "given": "Given", "find": "Find", "check": "Check", "result": "Result",
           "rec": "Lecture recording", "step": "Step", "steps": "steps", "page": "page", "slide": "slide",
           "intro": "Let us go through this question step by step."},
}


def video_cfg() -> dict:
    return {**DEFAULTS, **(get_config().get("video") or {})}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


def _t(board: dict) -> dict:
    return TEXT.get(board.get("language") or "en", TEXT["en"])


def _pages(board: dict) -> list[dict]:
    return board.get("pages") or ([board["page"]] if board.get("page") else [])


# --- 1. script: board -> scenes --------------------------------------------------------------------
def build_script(board: dict, question: str) -> list[dict]:
    """[{layout: intro | page | whiteboard, say, highlights, page_idx, steps_shown, points_shown,
    final, step, steps}], at most video.max_scenes (+ the title card)."""
    cfg = video_cfg()
    lang = board.get("language") or "en"
    scenes: list[dict] = []
    if board.get("view") == "whiteboard" and board.get("whiteboard"):
        wb = board["whiteboard"]
        steps = [s if isinstance(s, dict) else {"write": str(s), "say": ""} for s in wb["steps"]]
        for i, step in enumerate(steps, 1):
            scenes.append({"layout": "whiteboard", "steps_shown": i, "final": False,
                           "say": step.get("say") or speakable(step["write"], lang)})
        closing = " ".join(x for x in (wb.get("closing"), wb.get("check")) if x)
        if closing or wb.get("result"):
            scenes.append({"layout": "whiteboard", "steps_shown": len(steps), "final": True,
                           "say": closing or speakable(wb["result"], lang)})
    elif board.get("view") == "page" and _pages(board):
        lesson = board.get("lesson") or {}
        pages = _pages(board)
        beats = lesson.get("beats") or []
        for i, beat in enumerate(beats, 1):
            pi = int(beat.get("page") or 0)
            scenes.append({"layout": "page", "page_idx": pi if pi < len(pages) else 0,
                           "highlights": [beat["line"]] if beat.get("line") else [],
                           "points_shown": i, "say": beat["say"], "final": False})
        if lesson.get("closing") or not scenes:
            scenes.append({"layout": "page", "page_idx": 0, "highlights": pages[0].get("highlights") or [],
                           "points_shown": len(beats), "say": lesson.get("closing") or question,
                           "final": True})
    scenes = [s for s in scenes if (s["say"] or "").strip()][:int(cfg["max_scenes"])]
    for n, sc in enumerate(scenes, 1):
        sc["step"], sc["steps"] = n, len(scenes)
    if scenes and cfg.get("intro", True):
        scenes.insert(0, {"layout": "intro", "say": _t(board)["intro"], "final": False,
                          "step": 0, "steps": len(scenes)})
    return scenes


def split_subtitles(text: str, max_chars: int = 90) -> list[str]:
    """Cut what is said into pieces short enough for a two-line subtitle (sentence boundaries
    first, then commas / 'and', then words)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    pieces: list[str] = []
    for s in sentences:
        while len(s) > max_chars:   # one long sentence: cut at a comma / word near the middle
            cut = -1
            for pat in (r",\s", r"\s(?:ve|and|ama|but|çünkü|because|yani|which)\s", r"\s"):
                spots = [m.start() for m in re.finditer(pat, s[:max_chars]) if m.start() > max_chars * 0.35]
                if spots:
                    cut = spots[-1] + (1 if pat == r",\s" else 0)
                    break
            if cut <= 0:
                cut = max_chars
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            if pieces and len(pieces[-1]) + 1 + len(s) <= max_chars and not pieces[-1].endswith(","):
                pieces[-1] = f"{pieces[-1]} {s}"
            else:
                pieces.append(s)
    return [p for p in pieces if p]


# --- 2. pictures -----------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"   # full Turkish character set
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def _wrap(draw, text: str, font, width: int, max_lines: int) -> list[str]:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;") + " …"
    return lines


@lru_cache(maxsize=256)
def _formula_image(text: str, size: int = 30, color=BOARD_INK):
    """One line with formulas as an RGBA image (matplotlib mathtext in the Computer Modern look,
    i.e. like LaTeX). Formulas the model forgot to put in $...$ are fixed first; anything
    mathtext cannot draw falls back to plain words."""
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def render(s: str):
        fig = Figure(figsize=(0.01, 0.01), dpi=100)
        FigureCanvasAgg(fig)
        fig.text(0, 0, s, fontsize=size, color=tuple(c / 255 for c in color), family="DejaVu Sans",
                 math_fontfamily="cm")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")

    try:
        return render(for_mathtext(text))
    except Exception:  # noqa: BLE001  (LaTeX mathtext does not know)
        try:
            parts = for_mathtext(text).split("$")   # draw the formulas one by one, as text if needed
            return render("".join(p if i % 2 == 0 else speakable(p) for i, p in enumerate(parts)))
        except Exception:  # noqa: BLE001
            return render(speakable(text).replace("$", r"\$"))


class _Canvas:
    """A frame in 1280x720 design units, drawn at the real video size."""

    def __init__(self, cfg: dict):
        from PIL import Image, ImageDraw
        self.W, self.H = int(cfg["width"]) // 2 * 2, int(cfg["height"]) // 2 * 2
        self.k = self.W / 1280
        self.im = Image.new("RGB", (self.W, self.H), BG)
        self.d = ImageDraw.Draw(self.im)

    def s(self, v: float) -> int:
        return int(round(v * self.k))

    def box(self, x0, y0, x1, y1) -> list[int]:
        return [self.s(x0), self.s(y0), self.s(x1), self.s(y1)]

    def font(self, size: int, bold: bool = False):
        return _font(max(8, self.s(size)), bold)

    def text(self, x, y, text, size, color=INK, bold=False):
        self.d.text((self.s(x), self.s(y)), text, font=self.font(size, bold), fill=color)

    def wrap(self, text, size, width, max_lines, bold=False) -> list[str]:
        return _wrap(self.d, text, self.font(size, bold), self.s(width), max_lines)

    def textlength(self, text, size, bold=False) -> float:
        return self.d.textlength(text, font=self.font(size, bold)) / self.k


def _pip_box(cfg: dict) -> tuple[int, int, int, int]:
    """Where the lecture recording sits (x, y, w, h) in video pixels; that corner is left free."""
    W, H = int(cfg["width"]) // 2 * 2, int(cfg["height"]) // 2 * 2
    k = W / 1280
    w = int(float(cfg["pip_width"]) * k) // 2 * 2
    h = int(w * 9 / 16) // 2 * 2
    m = int(24 * k)
    return W - w - m, H - h - m, w, h


# layout (design units)
LEFT_X1 = 840         # main picture: x 24..840
PANEL_X0 = 856        # side panel: x 856..1256
TOP = 84
SUB_TOP = 590         # subtitle box: y 590..702 (under the main picture)


def _header(c: _Canvas, board: dict, title: str, scene: dict) -> None:
    t = _t(board)
    c.d.rectangle(c.box(0, 0, 1280, 64), fill=PANEL)
    c.text(28, 18, t["label"], 24, ACCENT, bold=True)
    x = 28 + c.textlength(t["label"], 24, True) + 18
    right = 140 if scene.get("step") else 40
    line = c.wrap(title, 22, 1280 - x - right, 1)
    if line:
        c.text(x, 20, line[0], 22)
    if scene.get("step"):
        label = f"{t['step']} {scene['step']}/{scene['steps']}"
        w = c.textlength(label, 20, True)
        c.d.rounded_rectangle(c.box(1256 - w - 28, 16, 1256, 48), radius=c.s(16), fill=PANEL2)
        c.text(1256 - w - 14, 20, label, 20, ACCENT, bold=True)


def _subtitle(c: _Canvas, text: str, x1: float) -> None:
    lines = c.wrap(text, 27, x1 - 24 - 44, 2)
    if not lines:
        return
    bottom = 702
    top = bottom - 22 - 38 * len(lines)
    c.d.rounded_rectangle(c.box(24, top, x1, bottom), radius=c.s(14), fill=PANEL, outline=ACCENT, width=c.s(2))
    for i, line in enumerate(lines):
        c.text(46, top + 12 + 38 * i, line, 27)


def _panel(c: _Canvas, x0, y0, x1, y1, title: str) -> float:
    c.d.rounded_rectangle(c.box(x0, y0, x1, y1), radius=c.s(14), fill=PANEL)
    c.text(x0 + 20, y0 + 16, title, 21, ACCENT, bold=True)
    return y0 + 54


def _bullets(c: _Canvas, items: list[str], x0, y, x1, y_max, current: int = -1, size: int = 20) -> float:
    for i, item in enumerate(items):
        lines = c.wrap(item, size, x1 - x0 - 34, 3)
        need = len(lines) * (size + 8) + 10
        if y + need > y_max:
            break
        color = ACCENT if i == current else INK
        c.d.ellipse(c.box(x0, y + size * 0.35, x0 + 9, y + size * 0.35 + 9), fill=color)
        for j, line in enumerate(lines):
            c.text(x0 + 22, y + j * (size + 8), line, size, color, bold=(i == current))
        y += need
    return y


def _paste_formula(c: _Canvas, text: str, x, y, max_w, size, color) -> int:
    """Draw a formula line; returns its height in design units."""
    img = _formula_image(text, c.s(size), color)
    if img.width > c.s(max_w):
        img = img.resize((c.s(max_w), max(1, int(img.height * c.s(max_w) / img.width))))
    c.im.paste(img, (c.s(x), c.s(y)), img)
    return int(img.height / c.k) + 1


def _fit_page(page, box, aw: int, ah: int, content=None):
    """Crop the page around the explained lines (zoom in like stepping up to the board), never
    narrower than the page's text (`content` box), and return (image, crop_x0, crop_y0, w, h)."""
    W, H = page.width, page.height
    if not box:
        return page, 0, 0, W, H
    ar = aw / ah
    bw, bh = box[2] - box[0], box[3] - box[1]
    cw_min = (content[2] - content[0] + 0.05) if content else 0.0
    f = min(1.0, max(0.62, bw + 0.16, cw_min, (bh + 0.22) * H * ar / W))
    cw = W * f
    ch = cw / ar
    if ch > H:
        ch = H
        cw = min(W, ch * ar)
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    if content and cw >= (content[2] - content[0]) * W:   # keep the text lines whole
        cx = (content[0] + content[2]) / 2 * W
    x0 = min(max(0, cx - cw / 2), W - cw)
    y0 = min(max(0, cy - ch / 2), H - ch)
    return page.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))), x0, y0, cw, ch


def _page_scene(c: _Canvas, scene: dict, board: dict, page_png, has_pip: bool, pip_top: float) -> None:
    from PIL import Image
    t = _t(board)
    pages = _pages(board)
    pi = scene.get("page_idx", 0)
    meta = pages[pi] if pi < len(pages) else pages[0]
    page = Image.open(io.BytesIO(page_png(scene.get("highlights") or [], pi))).convert("RGB")
    x0, y0, x1, y1 = 24, TOP, LEFT_X1, SUB_TOP - 40
    aw, ah = c.s(x1 - x0), c.s(y1 - y0)
    box = scene.get("box")
    shown, cx0, cy0, cw, ch = _fit_page(page, box, aw, ah, scene.get("content"))
    scale = min(aw / shown.width, ah / shown.height)
    shown = shown.resize((max(1, int(shown.width * scale)), max(1, int(shown.height * scale))))
    ox = c.s(x0) + (aw - shown.width) // 2
    oy = c.s(y0) + (ah - shown.height) // 2
    c.d.rounded_rectangle([ox - c.s(6), oy - c.s(6), ox + shown.width + c.s(6), oy + shown.height + c.s(6)],
                          radius=c.s(10), fill=PANEL2)
    c.im.paste(shown, (ox, oy))
    if box and scene.get("highlights"):   # frame the line being explained and point at it
        k = scale
        bx0 = ox + (box[0] * page.width - cx0) * k
        by0 = oy + (box[1] * page.height - cy0) * k
        bx1 = ox + (box[2] * page.width - cx0) * k
        by1 = oy + (box[3] * page.height - cy0) * k
        pad = c.s(7)
        c.d.rounded_rectangle([bx0 - pad, by0 - pad, bx1 + pad, by1 + pad], radius=c.s(8),
                              outline=MARK, width=max(2, c.s(3)))
        my = (by0 + by1) / 2
        ax = max(ox + c.s(4), bx0 - pad - c.s(30))
        c.d.polygon([(ax, my - c.s(11)), (ax + c.s(22), my), (ax, my + c.s(11))], fill=MARK)
    kind = t["slide"] if meta.get("type") == "pptx" else t["page"]
    c.text(28, SUB_TOP - 32, f"{meta['document']} · {kind} {meta['page']}", 16, MUTED)

    # side panel: key points so far (the closing scene: the summary)
    lesson = board.get("lesson") or {}
    beats = lesson.get("beats") or []
    bottom = pip_top - 46 if has_pip else 702
    if scene.get("final") and lesson.get("summary"):
        y = _panel(c, PANEL_X0, TOP, 1256, bottom, t["summary"])
        _bullets(c, [plain_sentence(x) for x in lesson["summary"]], PANEL_X0 + 20, y, 1256 - 16, bottom - 10)
    else:
        points = [b.get("point") or "" for b in beats][:scene.get("points_shown", 0)]
        points = [plain_sentence(p) for p in points if p]
        y = _panel(c, PANEL_X0, TOP, 1256, bottom, t["points"])
        if points:
            cur = -1 if scene.get("final") else len(points) - 1
            items = points[-8:]
            _bullets(c, items, PANEL_X0 + 20, y, 1256 - 16, bottom - 10, current=-1 if cur < 0 else len(items) - 1)


def _whiteboard_scene(c: _Canvas, scene: dict, board: dict, has_pip: bool, pip_top: float) -> None:
    t = _t(board)
    wb = board["whiteboard"]
    x0, y0, x1, y1 = 24, TOP, LEFT_X1, SUB_TOP - 14
    c.d.rounded_rectangle(c.box(x0, y0, x1, y1), radius=c.s(16), fill=BOARD_BG, outline=(214, 214, 207),
                          width=c.s(3))
    y = y0 + 16
    if wb.get("title"):
        c.text(x0 + 28, y, c.wrap(wb["title"], 28, x1 - x0 - 56, 1, True)[0], 28, BOARD_TITLE, bold=True)
        y += 50
    steps = [s if isinstance(s, dict) else {"write": str(s)} for s in wb["steps"]][:scene["steps_shown"]]
    rows = [(f"{i}.", s["write"]) for i, s in enumerate(steps, 1)]
    show_result = scene.get("final") and wb.get("result")
    images = [(n, _formula_image(txt, c.s(29), BOARD_INK)) for n, txt in rows]
    res_img = _formula_image(wb["result"], c.s(31), BOARD_TITLE) if show_result else None
    room = (y1 - 14 - y) - ((res_img.height / c.k + 34) if res_img is not None else 0)
    while images and sum(img.height / c.k + 12 for _, img in images) > room:
        images.pop(0)   # keep the newest lines on the board, like wiping the top
    maxw = c.s(x1 - x0 - 110)
    for k, (n, img) in enumerate(images):
        if img.width > maxw:
            img = img.resize((maxw, max(1, int(img.height * maxw / img.width))))
        h = img.height / c.k
        c.text(x0 + 28, y + max(0, h / 2 - 17), n, 27, BOARD_TITLE, bold=True)
        c.im.paste(img, (c.s(x0 + 72), c.s(y)), img)
        if k == len(images) - 1 and not show_result:   # the line just written
            c.d.line([c.s(x0 + 72), c.s(y + h + 3), c.s(x0 + 72) + img.width, c.s(y + h + 3)],
                     fill=ACCENT, width=max(2, c.s(4)))
        y += h + 12
    if res_img is not None:   # the result, framed
        img = res_img if res_img.width <= maxw else res_img.resize((maxw, max(1, int(res_img.height * maxw / res_img.width))))
        h = img.height / c.k
        y += 8
        c.d.rounded_rectangle([c.s(x0 + 56), c.s(y - 8), c.s(x0 + 88) + img.width, c.s(y + h + 8)],
                              radius=c.s(10), outline=ACCENT, width=max(2, c.s(3)), fill=(232, 248, 245))
        c.im.paste(img, (c.s(x0 + 72), c.s(y)), img)
        y += h + 16
        if wb.get("check") and y + 30 < y1:
            line = c.wrap(f"✓ {t['check']}: {plain_sentence(wb['check'])}", 18, x1 - x0 - 100, 2)
            for j, ln in enumerate(line):
                if y + 26 * (j + 1) < y1:
                    c.text(x0 + 60, y + 26 * j, ln, 18, BOARD_TITLE)

    # side panel: given / find (and the check at the end)
    bottom = pip_top - 46 if has_pip else 702
    y = _panel(c, PANEL_X0, TOP, 1256, bottom, t["given"])
    for g in (wb.get("given") or [])[:6]:
        if y + 34 > bottom - 120:
            break
        c.d.ellipse(c.box(PANEL_X0 + 20, y + 10, PANEL_X0 + 29, y + 19), fill=INK)
        y += _paste_formula(c, g, PANEL_X0 + 40, y, 1256 - PANEL_X0 - 60, 20, INK) + 8
    if wb.get("find") and y + 80 < bottom:
        y += 10
        c.text(PANEL_X0 + 20, y, t["find"], 21, ACCENT, bold=True)
        y += 34
        y += _paste_formula(c, wb["find"], PANEL_X0 + 20, y, 1256 - PANEL_X0 - 40, 20, INK) + 8


def _intro_scene(c: _Canvas, board: dict, title: str, course: str) -> None:
    t = _t(board)
    c.d.rounded_rectangle(c.box(120, 150, 1160, 560), radius=c.s(22), fill=PANEL, outline=ACCENT, width=c.s(2))
    c.text(170, 190, f"{course} · {t['question']}", 24, ACCENT, bold=True)
    y = 250
    for line in c.wrap(title, 38, 940, 5, True):
        c.text(170, y, line, 38, INK, bold=True)
        y += 54
    lesson = board.get("lesson") or {}
    n = len(lesson.get("beats") or []) or len((board.get("whiteboard") or {}).get("steps") or [])
    c.text(170, 500, f"{t['label']} · {n} {t['steps']}" if n else t["label"], 22, MUTED)


def render_frame(scene: dict, board: dict, course: str, title: str, has_pip: bool,
                 page_png: Callable[..., bytes], cfg: Optional[dict] = None, subtitle: Optional[str] = None):
    """One PIL image for one scene (subtitle = the piece of the sentence being spoken)."""
    cfg = cfg or video_cfg()
    c = _Canvas(cfg)
    _header(c, board, title, scene)
    px, py, pw, ph = _pip_box(cfg)
    pip_top = py / c.k - 34
    if scene["layout"] == "intro":
        _intro_scene(c, board, title, course)
    elif scene["layout"] == "page":
        _page_scene(c, scene, board, page_png, has_pip, pip_top)
    else:
        _whiteboard_scene(c, scene, board, has_pip, pip_top)
    if has_pip and scene["layout"] != "intro":
        c.d.rounded_rectangle([px - c.s(4), py - c.s(34), px + pw + c.s(4), py + ph + c.s(4)],
                              radius=c.s(10), fill=PANEL)
        c.d.text((px + c.s(10), py - c.s(28)), _t(board)["rec"], font=c.font(18), fill=MUTED)
    text = plain_sentence(scene["say"]) if subtitle is None else subtitle
    if scene["layout"] != "intro":
        _subtitle(c, text, LEFT_X1)
    else:
        _subtitle_center(c, text)
    return c.im


def _subtitle_center(c: _Canvas, text: str) -> None:
    lines = c.wrap(text, 27, 1000, 2)
    y = 610
    for line in lines:
        w = c.textlength(line, 27)
        c.text(640 - w / 2, y, line, 27, INK)
        y += 38


# --- 3. sound + 4. video ---------------------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip().splitlines()[-1] if res.stderr else res.returncode}")


def compose(frames: list[Path], audios: list[Path], out: Path, pip: Optional[dict], cfg: dict,
            pip_from: float = 0.0) -> float:
    """Frames + audio -> MP4 (H.264 + AAC). pip = {path, start_s, duration_s} or None; the corner
    video starts at `pip_from` seconds (after the title card). Returns the length in seconds."""
    from backend.media.tts import wav_seconds
    ff = ffmpeg_exe()
    fps = str(int(cfg["fps"]))
    work = out.parent / f".{out.stem}_parts"
    work.mkdir(parents=True, exist_ok=True)
    try:
        segs = []
        for i, (fr, au) in enumerate(zip(frames, audios)):
            seg = work / f"seg_{i:03d}.mp4"
            _run([ff, "-y", "-loop", "1", "-framerate", fps, "-i", str(fr), "-i", str(au),
                  "-t", f"{wav_seconds(au):.3f}", "-c:v", "libx264", "-preset", "veryfast", "-tune",
                  "stillimage", "-crf", "18", "-pix_fmt", "yuv420p", "-r", fps, "-c:a", "aac", "-b:a", "128k",
                  "-ar", "44100", "-ac", "1", str(seg)])
            segs.append(seg)
        listing = work / "list.txt"
        listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
        lesson = work / "lesson.mp4"
        _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(lesson)])
        total = sum(wav_seconds(a) for a in audios)
        fade = (f"fade=t=in:st=0:d=0.4,fade=t=out:st={max(0.0, total - 0.6):.2f}:d=0.6")
        afade = f"afade=t=in:st=0:d=0.3,afade=t=out:st={max(0.0, total - 0.6):.2f}:d=0.6"
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", fps,
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
        if pip:
            px, py, pw, ph = _pip_box(cfg)
            dur = max(1.0, min(float(pip["duration_s"]), total - pip_from))
            _run([ff, "-y", "-i", str(lesson), "-ss", f"{pip['start_s']:.2f}", "-t", f"{dur:.2f}",
                  "-i", str(pip["path"]),
                  "-filter_complex",
                  f"[1:v]scale={pw}:{ph}:force_original_aspect_ratio=decrease,"
                  f"pad={pw}:{ph}:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS+{pip_from:.3f}/TB[pip];"
                  f"[0:v][pip]overlay={px}:{py}:eof_action=pass,{fade}[v];[0:a]{afade}[a]",
                  "-map", "[v]", "-map", "[a]", *enc, str(out)])
        else:
            _run([ff, "-y", "-i", str(lesson), "-vf", fade, "-af", afade, *enc, str(out)])
        return total
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_video(question: str, course: str, board: dict, out: Path,
               progress: Callable[[str, float], None] = lambda s, f: None) -> dict:
    """Build the MP4 at `out`. Returns {path, seconds, scenes, voice, pip, notes}."""
    from backend.media import tts
    from backend.tasks.board import highlight_box, render_page
    from backend.tasks.board import text_language
    cfg = video_cfg()
    notes: list[str] = []
    # the voice (and the labels) follow the language the explanation is actually written in, so a
    # Turkish voice never reads English sentences (or the other way round)
    spoken = " ".join(sc["say"] for sc in build_script(board, question) if sc["layout"] != "intro")
    if spoken.strip():
        actual = text_language(spoken)
        if actual != (board.get("language") or "en"):
            board = {**board, "language": actual}
    scenes = build_script(board, question)
    if not scenes:
        raise ValueError("This answer has no explanation to turn into a video.")

    pip = None
    v = board.get("video")
    if cfg["pip"] and v:
        from backend.tasks.study import upload_path
        try:
            pip = {"path": upload_path(course, v["document"]), "start_s": v["start_ms"] / 1000,
                   "duration_s": max(1.0, (v["end_ms"] - v["start_ms"]) / 1000)}
        except (FileNotFoundError, ValueError):
            notes.append("The lecture recording is not on the server, so it is not shown in the corner.")

    pages = _pages(board)
    pngs: dict = {}

    def page_png(lines: list, pi: int = 0) -> bytes:
        key = (pi, tuple(lines))
        if key not in pngs:
            p = pages[pi]
            pngs[key] = render_page(course, p["document"], p["page"], list(lines), zoom=2.4)
        return pngs[key]

    from backend.tasks.board import page_lines, pdf_for
    contents: dict = {}
    for sc in scenes:
        if sc["layout"] == "page" and sc.get("highlights"):
            pi = sc.get("page_idx", 0)
            p = pages[pi]
            try:
                sc["box"] = highlight_box(course, p["document"], p["page"], sc["highlights"])
                if pi not in contents:   # where the page's text is (the zoom keeps it whole)
                    contents[pi] = highlight_box(course, p["document"], p["page"],
                                                 page_lines(pdf_for(course, p["document"]), p["page"]))
                sc["content"] = contents[pi]
            except Exception:  # noqa: BLE001
                sc["box"] = None
    language = board.get("language") or "en"
    voice_problem = tts.available()
    max_chars = int(cfg["subtitle_chars"])
    with tempfile.TemporaryDirectory(prefix="lecvid_") as tmp:
        tmp = Path(tmp)
        frames, audios, intro_s = [], [], 0.0
        # formulas in a sentence: shown with symbols in the subtitle (x₂, ×, √), read out in words
        # by the voice ("x iki", "çarpı"); never "$" or LaTeX on screen or in the voice
        pieces = []
        for sc in scenes:
            masked, formulas = mask_math(sc["say"])
            chunks = split_subtitles(masked, max_chars) or [masked]
            for j, chunk in enumerate(chunks):
                pieces.append((sc, unmask(chunk, formulas, pretty),
                               unmask(chunk, formulas, lambda f: speakable(f"${f}$", language)),
                               j == len(chunks) - 1))
        for n, (sc, sub, _, _) in enumerate(pieces):
            progress("frames", 0.05 + 0.35 * n / len(pieces))
            fr = tmp / f"frame_{n:03d}.png"
            render_frame(sc, board, course, question, pip is not None, page_png, cfg, subtitle=sub).save(fr)
            frames.append(fr)
        for n, (sc, _, piece, last) in enumerate(pieces):
            progress("voice", 0.4 + 0.45 * n / len(pieces))
            au = tmp / f"say_{n:03d}.wav"
            pause = float(cfg["pause_s"]) if last else float(cfg["chunk_pause_s"])
            if voice_problem is None:
                try:
                    tts.synthesize(piece, language, au, pause_s=pause)
                except Exception as exc:  # noqa: BLE001
                    voice_problem = f"voice unavailable ({exc})"
            if voice_problem is not None:
                tts.silence(float(cfg["seconds_per_word"]) * len(piece.split()) + pause + 0.6, au)
            audios.append(au)
            if sc["layout"] == "intro":
                intro_s += tts.wav_seconds(au)
        if voice_problem:
            notes.append(f"No narration voice: {voice_problem}. The video has subtitles only; "
                         "ask again (or press 'Record again') to retry the voice.")
        progress("video", 0.88)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_suffix(".part.mp4")
        seconds = compose(frames, audios, tmp_out, pip, cfg, pip_from=intro_s)
        tmp_out.replace(out)
    progress("done", 1.0)
    return {"path": str(out), "seconds": round(seconds, 1), "scenes": len(scenes),
            "voice": None if voice_problem else tts.voice_name(language), "pip": pip is not None,
            "notes": notes}


# --- background jobs -------------------------------------------------------------------------------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_WORKER = threading.Semaphore(1)   # one video at a time (CPU-heavy)


def job_id(question: str, course: str, board: dict) -> str:
    from backend.media.tts import voice_name
    key = {"v": VERSION, "q": question, "c": course, "board": board,
           "voice": voice_name(board.get("language") or "en"), "cfg": video_cfg()}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def output_path(course: str, jid: str) -> Path:
    from backend.tasks.study import upload_path  # noqa: F401  (same course-name rules)
    if not course or Path(course).name != course:
        raise ValueError("Invalid course")
    return resolve_path(video_cfg()["output_dir"]) / course / f"{jid}.mp4"


def _silent_path(out: Path) -> Path:
    return out.with_name(out.stem + ".silent.mp4")


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "state", "stage", "progress", "error", "seconds",
                                     "scenes", "voice", "pip", "notes")}


def start(question: str, course: str, board: dict) -> dict:
    """Start (or reuse) the video for this answer. Returns the job status immediately.
    A finished video with a voice is reused; one made without a voice is built again (the voice
    may work now)."""
    cfg = video_cfg()
    if not cfg["enabled"]:
        raise ValueError("Video generation is disabled (video.enabled in config.yaml).")
    if not build_script(board, question):
        raise ValueError("This answer has no explanation to turn into a video.")
    jid = job_id(question, course, board)
    out = output_path(course, jid)
    with _JOBS_LOCK:
        job = _JOBS.get(jid)
        if job and (job["state"] in ("queued", "running") or (job["state"] == "done" and job.get("voice"))):
            return _public(job)
        job = {"id": jid, "course": course, "state": "queued", "stage": "queued", "progress": 0.0,
               "error": None, "path": str(out), "notes": []}
        if out.is_file():  # built earlier, with a voice (cache)
            from backend.media.tts import voice_name
            job.update(state="done", stage="done", progress=1.0, voice=voice_name(board.get("language") or "en"))
            _JOBS[jid] = job
            return _public(job)
        _JOBS[jid] = job

    def progress(stage: str, frac: float) -> None:
        job.update(stage=stage, progress=round(frac, 2))

    def work():
        with _WORKER:
            job.update(state="running", stage="frames")
            try:
                tmp = out.with_name(out.stem + ".build.mp4")
                res = make_video(question, course, board, tmp, progress)
                final = out if res["voice"] else _silent_path(out)
                tmp.replace(final)
                job.update(state="done", stage="done", progress=1.0, path=str(final),
                           **{k: res[k] for k in ("seconds", "scenes", "voice", "pip", "notes")})
            except Exception as exc:  # noqa: BLE001
                log.warning("Lecture video failed: %s", exc)
                job.update(state="error", error=str(exc))

    threading.Thread(target=work, daemon=True, name=f"lecvid-{jid}").start()
    return _public(job)


def status(jid: str) -> dict:
    job = _JOBS.get(jid)
    if job is None:
        raise LookupError("Unknown video (the backend may have restarted; generate it again).")
    return _public(job)


def file_for(jid: str) -> Path:
    job = _JOBS.get(jid)
    if job is None or job["state"] != "done" or not Path(job["path"]).is_file():
        raise LookupError("The video is not ready.")
    return Path(job["path"])


def wait(jid: str, timeout: float = 120.0) -> dict:
    """Block until the job ends (tests / scripts)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = status(jid)
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.1)
    raise TimeoutError(jid)
