"""Teaching board for the Q&A page (backend: POST /board, GET /board/page).

The answer is explained the way the lecturer would do it in class:
- answer from a lecture video -> the video plays small in the bottom-right corner from that second;
- the slide / page being talked about is in the middle (the lesson may move on to the next cited
  page), and the lecturer explains it in spoken steps (speech bubble, words appearing as if
  spoken), highlighting the line being explained and collecting the key points; a summary closes;
- only a calculation / derivation / proof question (math, physics, ...) opens the whiteboard: what
  is given and asked, then the solution line by line while the lecturer says why, the framed result
  and a quick check.
The first time an answer is shown the explanation plays step by step; "Explain again" replays it.
The same explanation is also recorded as an MP4 (Piper voice, the slides / board, the lecture
recording in the corner) in the background; the player appears when it is ready.
"""
import html
import re
import time

import streamlit as st

from core import api_client as api
from core.state import mmss

WORD_DELAY_S = 0.045   # speech: one word after the other
BEAT_PAUSE_S = 0.7     # pause after each step, before the next line is pointed at / written

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap');
.st-key-corner_video{ position:fixed; right:24px; bottom:128px; width:340px; z-index:1000; background:#0b0f14;
  border:1px solid rgba(45,212,191,.5); border-radius:14px; overflow:hidden; box-shadow:0 14px 40px rgba(0,0,0,.5);
  gap:0 !important; }
.st-key-corner_video video{ width:100%; display:block; background:#000; }
.st-key-corner_video .mv-head{ font-size:.78rem; padding:.4rem .7rem; color:#b6c2d1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.st-key-corner_close{ position:absolute; top:2px; right:4px; z-index:2; }
.st-key-corner_close button{ min-height:0 !important; padding:.05rem .45rem !important; border:0 !important;
  background:rgba(0,0,0,.45) !important; color:#fff !important; }
@media (max-width: 760px){ .st-key-corner_video{ width:200px; right:12px; bottom:112px; } }
[class*="st-key-wb_"]{ background:#fbfbf8; border:1px solid #d6d6cf; border-radius:14px; padding:1.3rem 1.6rem 1rem;
  box-shadow: inset 0 0 0 6px #efefe9; min-height:220px; }
/* handwriting for the words only: KaTeX keeps its own fonts, or the symbols break */
[class*="st-key-wb_"] p, [class*="st-key-wb_"] li, [class*="st-key-wb_"] strong{
  color:#1d2330 !important; font-family:'Caveat', 'Segoe Print', cursive; font-size:1.45rem; line-height:1.5; }
[class*="st-key-wb_"] .katex{ color:#1d2330; font-size:1.0em; font-family:KaTeX_Main, 'Times New Roman', serif; }
[class*="st-key-wb_"] .katex-display{ margin:.3rem 0; }
[class*="st-key-wb_"] h4{ color:#0f5c55 !important; font-family:'Caveat', cursive; font-size:1.8rem; margin:0 0 .4rem; }
[class*="st-key-wbres_"]{ border:2px solid #0f5c55; border-radius:10px; background:#e8f8f5; padding:.2rem .8rem; margin:.4rem 0; }
[class*="st-key-wbres_"] p, [class*="st-key-wbres_"] .katex{ color:#0f5c55 !important; }
[class*="st-key-given_"]{ background:rgba(45,212,191,.06); border:1px dashed rgba(45,212,191,.45); border-radius:12px; padding:.6rem .9rem; }
[class*="st-key-given_"] p{ margin-bottom:.15rem; }
[class*="st-key-say_"]{ background:rgba(45,212,191,.08); border:1px solid rgba(45,212,191,.28); border-left:3px solid #2dd4bf;
  border-radius:12px; padding:.55rem .8rem; }
[class*="st-key-say_"] p{ margin-bottom:.2rem; }
</style>
"""


def _display_math(body: str) -> str:
    body = re.sub(r"\n\s*\n", "\n", body.strip())   # a blank line would end the block
    return "\n\n$$\n" + body + "\n$$\n\n"


def fix_markdown(text: str) -> str:
    r"""Make LLM math render in Streamlit: \( ... \) -> $...$, and every display formula
    (\[ ... \] or $$ ... $$) on lines of its own, which Streamlit needs to draw it with KaTeX
    (otherwise the LaTeX shows as text or as a red error)."""
    t = text or ""
    t = re.sub(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", lambda m: "$$" + m.group(1) + "$$", t, flags=re.S)
    t = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", t, flags=re.S)
    t = re.sub(r"\$\$(.+?)\$\$", lambda m: _display_math(m.group(1)), t, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _tr(b: dict) -> bool:
    return (b or {}).get("language") == "tr"


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=128)
def _page_png(course: str, document: str, page: int, highlights: tuple) -> bytes:
    return api.board_page_image(course, document, page, list(highlights))


def fetch(question: str, answer: str, course: str, lecture, sources: list, provider) -> dict:
    """Ask the backend what to show. Never raises: a board is a bonus, the answer stands alone."""
    try:
        return api.board(question, answer, course, lecture, sources, provider)
    except api.ApiError as exc:
        return {"view": "none", "video": None, "page": None, "whiteboard": None, "lesson": None,
                "notes": [f"Teaching board unavailable: {exc}"]}


def _close_corner() -> None:
    st.session_state.board_focus = None


def _corner_video(course: str, video: dict) -> None:
    start, end = video["start_ms"] // 1000, max(video["end_ms"] // 1000, video["start_ms"] // 1000 + 1)
    url = html.escape(f"{api.study_file_url(course, video['document'])}#t={start},{end}", quote=True)
    label = html.escape(f"🎬 {video['document']} · {mmss(video['start_ms'])}–{mmss(video['end_ms'])}")
    with st.container(key="corner_video"):   # fixed in the corner by the CSS above
        st.button("✕", key="corner_close", on_click=_close_corner, help="Close the video")
        st.markdown(f'<div class="mv-head">{label}</div>'
                    f'<video src="{url}" controls autoplay muted playsinline preload="auto"></video>',
                    unsafe_allow_html=True)


def _speech(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(WORD_DELAY_S)


def _say(key: str, text: str, animate: bool, line: str = "") -> None:
    """One speech bubble of the lecturer (words appear one by one the first time)."""
    if not text:
        return
    with st.container(key=key):
        if animate:
            st.write_stream(_speech(f"🧑‍🏫 {text}"))
        else:
            st.markdown(f"🧑‍🏫 {text}")
        if line:
            st.caption(f"📍 “{line}”")
    if animate:
        time.sleep(BEAT_PAUSE_S)


# --- whiteboard: the lecturer writes the solution and says why --------------------------------------
def _whiteboard(mid: str, wb: dict, animate: bool, tr: bool = False) -> None:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        board = st.container(key=f"wb_{mid}")
    with right:
        if wb.get("given") or wb.get("find"):
            with st.container(key=f"given_{mid}"):
                if wb.get("given"):
                    st.markdown("**" + ("Verilenler" if tr else "Given") + "**")
                    for g in wb["given"]:
                        st.markdown(f"• {g}")
                if wb.get("find"):
                    st.markdown("**" + ("İstenen" if tr else "Find") + f":** {wb['find']}")
        st.caption("Hocanın anlattıkları" if tr else "What the lecturer says")
    with board:
        if wb.get("title"):
            st.markdown(f"#### {wb['title']}")
    for i, step in enumerate(wb["steps"], 1):
        if isinstance(step, str):   # older boards: plain lines
            step = {"write": step, "say": ""}
        with board:
            st.markdown(f"**{i}.** {step['write']}")
        with right:
            _say(f"say_{mid}_w{i}", step.get("say", ""), animate)
        if animate and not step.get("say"):
            time.sleep(BEAT_PAUSE_S)
    with board:
        if wb.get("result"):
            with st.container(key=f"wbres_{mid}"):
                st.markdown(f"**{'Sonuç' if tr else 'Result'}:** {wb['result']}")
        if wb.get("check"):
            st.markdown(f"✓ *{'Kontrol' if tr else 'Check'}: {wb['check']}*")
    with right:
        _say(f"say_{mid}_wend", wb.get("closing", ""), animate)


# --- page / slide: the lecturer points at the lines one by one --------------------------------------
def _page_image(target, course: str, page: dict, lines: list) -> bool:
    try:
        png = _page_png(course, page["document"], page["page"], tuple(lines))
    except api.ApiError as exc:
        target.caption(f"The page cannot be shown: {exc}")
        return False
    target.image(png, width="stretch")
    return True


def _page_lesson(mid: str, course: str, b: dict, animate: bool) -> None:
    pages = b.get("pages") or [b["page"]]
    lesson, video, tr = b.get("lesson") or {}, b.get("video"), _tr(b)

    def caption(pi: int) -> str:
        page = pages[pi]
        kind = ("slayt" if tr else "slide") if page["type"] == "pptx" else ("sayfa" if tr else "page")
        if pi == 0 and page.get("reason") == "video-match" and video:
            why = (f"videoda {mmss(video['start_ms'])}'de anlatılan {kind}" if tr else
                   f"the {kind} that matches what is said at {mmss(video['start_ms'])} in the video")
        else:
            why = "kaynak" if tr else "cited"
        more = f" · {pi + 1}/{len(pages)}" if len(pages) > 1 else ""
        return f"📄 {page['document']}, {kind} {page['page']} — {why}{more}"

    left, right = st.columns([3, 2], gap="medium")
    with left:
        picture = st.empty()
        cap = st.empty()
        points_box = st.empty()
    with right:
        st.caption("Hocanın anlattıkları" if tr else "What the lecturer says")
    beats = lesson.get("beats") or []

    def show(pi: int, lines: list) -> None:
        _page_image(picture, course, pages[pi], lines)
        cap.caption(caption(pi))

    def points(upto: int) -> None:
        pts = [bt.get("point") for bt in beats[:upto] if bt.get("point")]
        if pts:
            points_box.markdown(f"**📌 {'Anahtar noktalar' if tr else 'Key points'}**\n\n"
                                + "\n".join(f"- {p}" for p in pts))

    if animate:
        show(0, [])
        for i, beat in enumerate(beats, 1):
            pi = min(int(beat.get("page") or 0), len(pages) - 1)
            show(pi, [beat["line"]] if beat.get("line") else [])   # point at the line being explained
            points(i)
            with right:
                _say(f"say_{mid}_p{i}", beat["say"], True, beat.get("line") or "")
    else:
        with right:
            for i, beat in enumerate(beats, 1):
                _say(f"say_{mid}_p{i}", beat["say"], False, beat.get("line") or "")
        points(len(beats))
    show(0, pages[0].get("highlights") or [])   # all the lines pointed at on the first page
    with right:
        _say(f"say_{mid}_pend", lesson.get("closing", ""), animate)
        if lesson.get("summary"):
            st.markdown(f"**🧾 {'Özet' if tr else 'Summary'}**\n\n" + "\n".join(f"- {x}" for x in lesson["summary"]))
    if len(pages) > 1:
        with st.expander("📄 " + ("Anlatımdaki diğer sayfalar" if tr else "Other pages in this explanation")):
            for pi in range(1, len(pages)):
                st.caption(caption(pi))
                _page_image(st, course, pages[pi], pages[pi].get("highlights") or [])


# --- the generated explanation video -------------------------------------------------------------
_STAGES = {"queued": "waiting", "frames": "drawing the slides / board", "voice": "recording the voice",
           "video": "editing the video", "done": "done"}


def start_video(question: str, course: str, board_result: dict) -> dict:
    """Ask the backend to record the explanation video (returns at once; it builds in the background)."""
    try:
        return api.start_lecture_video(question, course, board_result)
    except api.ApiError as exc:
        return {"id": None, "state": "error", "error": str(exc)}


@st.fragment(run_every=2)
def _video_progress(m: dict) -> None:
    job = m["video_job"]
    try:
        job = api.lecture_video_status(job["id"])
    except api.ApiError as exc:
        job = {**job, "state": "error", "error": str(exc)}
    m["video_job"] = job
    if job["state"] in ("done", "error"):
        st.rerun()   # the whole page, so the player replaces this progress bar
    st.progress(float(job.get("progress") or 0.0),
                text=f"🎞️ Recording the explanation video… {_STAGES.get(job.get('stage'), '')}")


def _video(m: dict) -> None:
    job = m.get("video_job")
    if not job:
        return
    st.markdown("**🎞️ Explanation video**")
    if job["state"] == "done":
        st.video(api.lecture_video_url(job["id"]))
        if not job.get("voice") and m.get("board") and st.button(
                "🔁 Record again with voice", key=f"vid_voice_{m['mid']}",
                help="The voice could not be downloaded last time; this tries again."):
            m["video_job"] = start_video(m["question"], m["course"], m["board"])
            st.rerun()
        bits = []
        if job.get("seconds"):
            bits.append(f"{job['seconds']:.0f} s")
        if job.get("voice"):
            bits.append(f"voice: {job['voice']} (Piper, local)")
        if job.get("pip"):
            bits.append("lecture recording in the corner")
        if bits:
            st.caption(" · ".join(bits))
        for note in job.get("notes") or []:
            st.caption(f"ℹ️ {note}")
    elif job["state"] == "error":
        st.caption(f"The video could not be made: {job.get('error')}")
        if "No module named" in str(job.get("error")):
            st.caption("A package of requirements.txt is missing on the server: run "
                       "`pip install -r requirements.txt` and restart uvicorn.")
        if m.get("board") and st.button("🔁 Try again", key=f"vid_retry_{m['mid']}"):
            m["video_job"] = start_video(m["question"], m["course"], m["board"])
            st.rerun()
    else:
        _video_progress(m)


def render(m: dict, course: str) -> None:
    """Draw the board of one assistant message (m holds the /ask reply plus m['board'])."""
    b = m.get("board")
    if not b or (b["view"] == "none" and not b.get("video")):
        return
    ss = st.session_state
    ss.setdefault("board_shown", set())
    mid = m["mid"]
    animate = mid not in ss.board_shown
    with st.container(border=True, key=f"board_{mid}"):
        head, btn = st.columns([4, 1])
        head.markdown("**🧑‍🏫 The lecturer explains**")
        if not animate and b["view"] in ("whiteboard", "page"):
            if btn.button("🔁 Explain again", key=f"again_{mid}"):
                ss.board_shown.discard(mid)
                if b.get("video"):
                    ss.board_focus = mid
                st.rerun()
        if b.get("video"):
            v = b["video"]
            st.caption(f"🎬 From the lecture video at {mmss(v['start_ms'])}–{mmss(v['end_ms'])}"
                       + (" · playing in the corner" if ss.get("board_focus") == mid else ""))
        if b["view"] == "whiteboard":
            _whiteboard(mid, b["whiteboard"], animate, _tr(b))
            if b.get("page") and b["page"].get("renderable"):
                with st.expander("📄 Related page in the course material"):
                    _page_image(st, course, b["page"], b["page"].get("highlights") or [])
        elif b["view"] == "page":
            _page_lesson(mid, course, b, animate)
        ss.board_shown.add(mid)
        _video(m)
        if b.get("video"):
            v = b["video"]
            if ss.get("board_focus") == mid:
                if st.button("✖ Close video", key=f"close_{mid}"):
                    ss.board_focus = None
                    st.rerun()
            elif st.button(f"▶ Play {mmss(v['start_ms'])}–{mmss(v['end_ms'])} in the corner", key=f"play_{mid}"):
                ss.board_focus = mid
                st.rerun()
        for note in b.get("notes") or []:
            st.caption(f"ℹ️ {note}")


def corner(chat: list[dict]) -> None:
    """The one corner video (the focused answer's). Called once per page run, after the chat."""
    _inject_css()
    focus = st.session_state.get("board_focus")
    for m in chat:
        if m.get("mid") == focus and (m.get("board") or {}).get("video"):
            _corner_video(m["course"], m["board"]["video"])
            return
