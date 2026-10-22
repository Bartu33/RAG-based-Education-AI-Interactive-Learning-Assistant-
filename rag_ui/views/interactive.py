"""Interactive Learning: read one of your documents and ask the study buddy (chat bot) about it.

The buddy only searches the chunks of the open document (fast, and the answers stay on topic).
"""
import base64
import html
import re
from pathlib import Path
from time import perf_counter
from typing import Optional

import streamlit as st

from core import api_client as api
from core import styles
from core.board import fix_markdown
from core.state import BeyondStream, cached_courses, cached_documents, invalidate, mmss, safe, split_beyond
from core.video_player import moment, player

styles.kicker("STUDY MODE"); st.title("🧠 Interactive Learning")
# this page shows a document and a chat side by side, so it may use more of the screen
st.markdown("<style>.block-container{max-width:1500px !important;}</style>", unsafe_allow_html=True)

ICONS = {"pdf": "📕", "pptx": "📊", "docx": "📝", "video": "🎬"}
BOT_ICON = Path(__file__).resolve().parent.parent / "assets" / "chat-bot-icon.png"
BOT_AVATAR = str(BOT_ICON) if BOT_ICON.is_file() else "🤖"   # chat avatar of the study buddy
ss = st.session_state
ss.setdefault("buddy_open", False)
ss.setdefault("buddy_chats", {})   # "course|document" -> [messages]
ss.setdefault("iv_jobs", {})       # video uploads of this session: "course|document" -> job id
RUNNING = ("queued", "transcribing")


@st.cache_data(ttl=60, show_spinner=False)
def outline(course: str, document: str) -> dict:
    return api.study_document(course, document)


@st.cache_data(max_entries=48, show_spinner=False)
def page_png(course: str, document: str, page: int) -> bytes:
    return api.study_page_image(course, document, page, zoom=1.6)


@st.cache_resource(show_spinner=False)
def bot_icon_uri() -> str:
    """The bot picture as a data URI (a Streamlit button cannot hold an image, so CSS draws it)."""
    if not BOT_ICON.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(BOT_ICON.read_bytes()).decode()


def toggle_buddy():
    ss.buddy_open = not ss.buddy_open


def open_document(course: str, document: str):
    """Select this document in the pickers (must run before they are drawn)."""
    invalidate()
    outline.clear()
    ss.iv_course = course
    ss.pop(f"iv_week_{course}", None)   # back to "All weeks"
    ss[f"iv_doc_{course}_None"] = f"{course}|{document}"


# --- 0. a lecture video of your own: transcribed in the background, watch it right away -----------
@st.fragment(run_every=3)
def upload_progress():
    for k, job_id in list(ss.iv_jobs.items()):
        job = safe(api.video_job, job_id)
        if job is None:
            ss.iv_jobs.pop(k); continue
        name, total = job["document"], job["duration_ms"]
        if job["state"] == "failed":
            st.error(f"🎬 {name}: {job['error']}")
            if st.button("OK", key=f"jobok_{job_id}"):
                ss.iv_jobs.pop(k); st.rerun()
            continue
        done = job["state"] == "done"
        if done and ss.get("iv_current") == k:   # finished and open in the viewer: nothing to show
            ss.iv_jobs.pop(k); continue
        share = 1.0 if done else (job["done_ms"] / total if total else 0.0)
        text = (f"🎬 **{name}**: ready ({job['chunks']} transcript parts)" if done else
                f"🎬 **{name}**: transcribing… {mmss(job['done_ms'])}"
                + (f" / {mmss(total)}" if total else "") + " · you can watch it while this runs")
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.progress(share, text=text)
        if job["chunks"] and c2.button("▶ Open", key=f"jobopen_{job_id}", width="stretch"):
            ss.iv_open = (job["course"], name)   # applied before the pickers are drawn
            st.rerun(scope="app")


with st.expander("🎬 Upload a lecture video to study with"):
    known = [c["course"] for c in (safe(cached_courses) or [])]
    NEW = "➕ New course…"
    u1, u2 = st.columns(2)
    pick = u1.selectbox("Course", known + [NEW], key="vu_course")
    v_course = (u1.text_input("New course code", placeholder="e.g. CENG336", key="vu_new")
                .replace(" ", "").upper() if pick == NEW else pick)
    v_week = u2.number_input("Week / lecture number", 1, 30, 1, key="vu_week")
    video = st.file_uploader("Lecture video", type=["mp4", "mkv", "avi"], key="vu_file")
    ok = st.checkbox("I have the right to use this recording.", key="vu_ok")
    st.caption("The speech is transcribed on the server (faster-whisper) in the background. The video "
               "can be watched at once; the study buddy knows the part transcribed so far.")
    if st.button("Upload and transcribe", type="primary", disabled=not (video and ok and v_course)):
        with st.spinner("Uploading…"):
            job = safe(api.upload_video, video.name, video.getvalue(), v_course, int(v_week),
                       video.type or "application/octet-stream")
        if job:
            ss.iv_jobs[f"{job['course']}|{job['document']}"] = job["id"]
if ss.iv_jobs:
    upload_progress()


# --- 1. pick a document ---------------------------------------------------------------------------
if ss.get("iv_open"):
    open_document(*ss.pop("iv_open"))
docs = safe(cached_documents)
if docs is None:
    st.stop()
if not docs:
    st.info("You have no indexed documents yet. Upload a lecture video above, or course materials:")
    st.page_link("views/materials.py", label="Upload materials →")
    st.stop()

def natural(name: str) -> list:
    """Sort key: "2- Sampling" before "10- FFT"."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


# course -> week -> document, like the folders on the Materials page
by_key = {f"{d['course']}|{d['document']}": d for d in docs}
p1, p2, p3 = st.columns([1, 1, 3])
course = p1.selectbox("Course", sorted({d["course"] for d in docs}, key=natural), key="iv_course")
weeks = sorted({d["lecture"] for d in docs if d["course"] == course})
week = p2.selectbox("Week", [None] + weeks, key=f"iv_week_{course}",
                    format_func=lambda w: "All weeks" if w is None else f"Week {w}")
keys = sorted((k for k, d in by_key.items() if d["course"] == course and week in (None, d["lecture"])),
              key=lambda k: (by_key[k]["lecture"], natural(by_key[k]["document"])))
key = p3.selectbox(
    "Document", keys, key=f"iv_doc_{course}_{week}",
    format_func=lambda k: f"{ICONS.get(by_key[k]['type'], '📄')} {by_key[k]['document']}"
                          + (f" · week {by_key[k]['lecture']}" if week is None and len(weeks) > 1 else ""))
d = by_key[key]
ss.iv_current = key
info = safe(outline, d["course"], d["document"])
if info is None:
    st.stop()
kind, units = info["type"], info["units"]
nav = f"iv_pos_{key}"   # viewer position: PDF page number, or 1-based unit index otherwise
seek = f"iv_seek_{key}"  # video position in seconds
pdf_pages = info["pages"] if kind == "pdf" and info["has_file"] else None
last = pdf_pages or len(units)
# a video with its file is shown in the player, which reports the moment the student is at
playing = kind == "video" and info["has_file"]
vp_key, vp_seek = f"vp_{key}", f"vp_seek_{key}"


def move(step: int):
    ss[nav] = min(max(1, ss.get(nav, 1) + step), last)


def go_to(src: dict):
    """Chat source button → show that page / slide / section / moment in the viewer."""
    if kind == "video":
        if src.get("start_ms") is not None and playing:
            n = (ss.get(vp_seek) or {}).get("n", 0) + 1
            ss[vp_seek] = {"t": src["start_ms"] / 1000, "n": n}
        elif src.get("start_ms") is not None:
            ss[seek] = int(src["start_ms"] // 1000)
        return
    if pdf_pages:
        ss[nav] = int(src["page"] or 1)
        return
    for i, u in enumerate(units, 1):
        if (kind in ("pdf", "pptx") and u["page"] == src.get("page")) or \
           (kind == "docx" and u["section"] == src.get("section")):
            ss[nav] = i
            return


def current_location() -> dict:
    """Where the student is looking (sent with every chat request)."""
    if kind == "video":
        return {"time_ms": int((moment(vp_key) if playing else ss.get(seek, 0)) * 1000)}
    pos = int(ss.get(nav, 1))
    if pdf_pages:
        return {"page": pos}
    u = units[min(pos, len(units)) - 1]
    return {"page": u["page"], "section": u["section"]}


@st.fragment(run_every=3)
def transcript_progress(course: str, document: str):
    """While the open video is still being transcribed: how far the buddy can follow it."""
    job = safe(api.video_status, course, document)
    if not job or job["state"] not in RUNNING:
        outline.clear()
        st.rerun(scope="app")
    total = f" of {mmss(job['duration_ms'])}" if job["duration_ms"] else ""
    st.info(f"⏳ Still transcribing: the study buddy knows the video up to **{mmss(job['done_ms'])}**{total}. "
            "The rest is added automatically.")


# one element whether or not it shows something, so the player below never moves in the page
with st.container():
    if kind == "video":
        job_now = safe(api.video_status, d["course"], d["document"])
        if job_now and job_now["state"] in RUNNING:
            transcript_progress(d["course"], d["document"])


# --- floating chat-bot button (fixed bottom right, see styles.py) ---------------------------------
icon = bot_icon_uri()
if icon:
    st.markdown(f"<style>.st-key-buddy_fab button, .st-key-buddy_fab_open button"
                f"{{background-image:url('{icon}') !important;}}</style>", unsafe_allow_html=True)
st.button("Study Buddy" if icon else "🤖", key="buddy_fab_open" if ss.buddy_open else "buddy_fab",
          on_click=toggle_buddy,
          help="Close the study buddy" if ss.buddy_open else "Ask the study buddy about this document")

if playing:
    # The player stays at this place in the page in both layouts (only CSS moves it), so the video
    # is not reloaded and keeps playing. With the chat open it is a small player where the chat
    # button was; clicking it brings the big video back.
    mini = ss.buddy_open
    st.markdown("<style>.st-key-vplayer{position:fixed; right:1.6rem; bottom:1.6rem; z-index:1000;"
                "width:min(380px, 28vw);} .st-key-buddy_fab_open{display:none;}</style>" if mini else
                "<style></style>", unsafe_allow_html=True)
    with st.container(key="vplayer"):
        event = player(api.study_file_url(d["course"], d["document"]), vp_key,
                       "mini" if mini else "main", moment(vp_key), ss.get(vp_seek), f"🎬 {d['document']}",
                       max_height=480)
    if event and event["event"] in ("expand", "ask") and ss.buddy_open != (event["event"] == "ask"):
        ss.buddy_open = event["event"] == "ask"
        st.rerun()

if playing and ss.buddy_open:   # the chat is the main element; the right side is the player's
    panel, _ = st.columns([3, 2], gap="medium")
    viewer = None
else:
    viewer, panel = (st.columns([3, 2], gap="medium") if ss.buddy_open else (st.container(), None))


# --- 2. document viewer ---------------------------------------------------------------------------
def doc_page(title: str, text: str):
    st.markdown(f'<div class="doc-page"><div class="doc-page-title">{html.escape(title)}</div>'
                f'{html.escape(text)}</div>', unsafe_allow_html=True)


if viewer is not None:
    with viewer:
        st.caption(f"{info['chunks']} indexed chunks · course {info['course']} · week {info['lecture']}")
        if kind == "video":
            dur = max(1, int(max((u["end_ms"] or 0) for u in units) // 1000))
            if not playing:
                ss.setdefault(seek, 0)
                st.info("The original video is not on the server; the transcript is shown below.")
                st.slider("I'm at (seconds) — the buddy explains this moment", 0, dur, key=seek, step=5)
                st.caption(f"⏱️ {mmss(int(ss[seek]) * 1000)} / {mmss(dur * 1000)}")
            with st.expander(f"📜 Transcript ({len(units)} parts)"):
                for u in units:
                    st.markdown(f"**{u['label']}** — {u['text']}")
        else:
            ss.setdefault(nav, 1)
            ss[nav] = min(max(1, int(ss[nav])), last)
            unit_word = {"pdf": "Page", "pptx": "Slide", "docx": "Section"}[kind]
            c1, c2, c3, c4 = st.columns([1, 2, 1, 3], vertical_alignment="bottom")
            c1.button("◀", key=f"prev_{key}", on_click=move, args=(-1,), disabled=ss[nav] <= 1,
                      width="stretch")
            c2.number_input(f"{unit_word} (of {last})", 1, last, key=nav)
            c3.button("▶", key=f"next_{key}", on_click=move, args=(1,), disabled=ss[nav] >= last,
                      width="stretch")
            text_view = c4.toggle("Text view", key=f"txt_{key}", value=not pdf_pages,
                                  disabled=not pdf_pages,
                                  help="Show the indexed text instead of the page image")
            pos = int(ss[nav])
            with st.container(height=760, border=True, key="doc_viewer"):
                if pdf_pages and not text_view:
                    png = safe(page_png, d["course"], d["document"], pos)
                    if png:
                        st.image(png, width="stretch")
                elif pdf_pages:
                    found = [u for u in units if u["page"] == pos]
                    if found:
                        doc_page(found[0]["label"], found[0]["text"])
                    else:
                        st.info("No text was extracted from this page (image-only page?).")
                else:
                    u = units[pos - 1]
                    doc_page(u["label"], u["text"])
                    if kind in ("pptx", "docx"):
                        st.caption("The preview shows the slide / section text the assistant uses.")
            if info["has_file"] and kind in ("pptx", "docx"):
                st.link_button("⬇️ Open the original file", api.study_file_url(d["course"], d["document"]))
        if not ss.buddy_open:
            st.caption("💡 Click the assistant button (bottom right) to ask questions about this document.")


# --- 3. study buddy chat --------------------------------------------------------------------------
def ask_label() -> str:
    return {"pdf": "Explain page", "pptx": "Explain slide",
            "docx": "Explain section", "video": "Explain moment"}[kind]


def explain_prompt() -> str:
    """What the chat shows for the explain button, e.g. "Explain page 3."."""
    loc = current_location()
    if kind == "video":
        return f"Explain the moment {mmss(loc['time_ms'])}."
    if kind == "docx":
        return f"Explain the section “{loc['section']}”." if loc.get("section") else "Explain this section."
    return f"Explain {'slide' if kind == 'pptx' else 'page'} {loc['page']}."


def render_sources(m: dict, idx: int):
    srcs = m.get("sources") or []
    if not srcs:
        return
    with st.expander(f"📎 {len(srcs)} excerpt{'s' if len(srcs) != 1 else ''} used"):
        for s in srcs:
            a, b = st.columns([4, 1], vertical_alignment="center")
            a.markdown(f"**[{s['n']}] {s['label']}**")
            a.caption(s["text"][:220] + ("…" if len(s["text"]) > 220 else ""))
            b.button("Go", key=f"go_{key}_{idx}_{s['n']}", on_click=go_to, args=(s,),
                     help="Show this part in the viewer")


if panel is not None:
    chat = ss.buddy_chats.setdefault(key, [])
    with panel:
        with st.container(border=True, key="buddy_panel"):
            h1, h2, h3, *h4 = st.columns([3, 2, 1, 1] if playing else [3, 2, 1], vertical_alignment="center")
            if h4:
                h4[0].button("🎬", key="buddy_back", on_click=toggle_buddy, help="Back to the video")
            logo = f'<img src="{icon}" alt="">' if icon else "🤖 "
            h1.markdown(f'<div class="buddy-title">{logo}Study Buddy</div>'
                        f'<div class="buddy-sub">{html.escape(d["document"])}</div>',
                        unsafe_allow_html=True)
            lang = h2.segmented_control("Answer language", ["EN", "TR"], default="EN", required=True,
                                        key="buddy_lang", label_visibility="collapsed",
                                        help="Language of the answers")
            if h3.button("🗑️", key="buddy_clear", help="Clear this chat", disabled=not chat):
                chat.clear(); st.rerun()

            actions = [("summary", "📝 Summarize", "Summarize this document."),
                       ("explain", f"💡 {ask_label()}", explain_prompt()),
                       ("concepts", "🔑 Key concepts", "What are the key concepts?")]
            pending: Optional[tuple] = None
            a_cols = st.columns(len(actions))
            for col, (mode, label, prompt) in zip(a_cols, actions):
                if col.button(label, key=f"act_{mode}", width="stretch"):
                    pending = (mode, prompt, "")

            box = st.container(height=520, border=False, key="buddy_msgs", autoscroll=True)
            with box:
                if not chat:
                    st.markdown('<div class="buddy-empty">👋 Hi! I only read <b>this</b> document, so I '
                                'answer fast. Ask me anything about it, or use a button above.</div>',
                                unsafe_allow_html=True)
                for i, m in enumerate(chat):
                    with st.chat_message(m["role"], avatar=BOT_AVATAR if m["role"] == "assistant" else "🧑‍🎓"):
                        st.markdown(fix_markdown(m["content"]) if m["role"] == "assistant" else m["content"])
                        if m["role"] == "assistant":
                            if m.get("beyond"):
                                styles.beyond_box(m["beyond"], f"{key}_{i}", "document")
                            render_sources(m, i)
                            if m.get("timing"):
                                st.caption(m["timing"])

            q = st.chat_input("Ask about this document…", key="buddy_input")
            if q and q.strip():
                pending = ("ask", q.strip(), q.strip())

            if pending:
                mode, shown, question = pending
                body = {"course": d["course"], "document": d["document"], "mode": mode,
                        "question": question, "language": (lang or "EN").lower(),
                        "history": [{"role": m["role"], "content": m["content"]} for m in chat],
                        "provider": ss.cfg.get("provider"), "top_context": ss.cfg.get("study_top_k"),
                        **current_location()}
                meta: dict = {}
                with box:
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.markdown(shown)
                    with st.chat_message("assistant", avatar=BOT_AVATAR):
                        t0 = perf_counter()
                        try:
                            stream = BeyondStream(api.study_chat_stream(body, meta))
                            st.write_stream(stream)
                            answer = stream.raw
                        except api.ApiError as exc:
                            st.error(f"Backend error: {exc}")
                            answer = None
                if answer:
                    total = perf_counter() - t0
                    chat.append({"role": "user", "content": shown})
                    main, beyond = split_beyond(answer)
                    # only the document-based part goes back to the model as conversation history
                    chat.append({"role": "assistant", "content": main or answer, "beyond": beyond,
                                 "sources": meta.get("sources", []),
                                 "timing": f"⚡ search {meta.get('retrieval_ms', '–')} ms · "
                                           f"total {total:.1f} s"})
                    st.rerun()
