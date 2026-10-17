"""Interactive Learning: read one of your documents and ask the study buddy (chat bot) about it.

The buddy only searches the chunks of the open document (fast, and the answers stay on topic).
"""
import base64
import html
from pathlib import Path
from time import perf_counter
from typing import Optional

import streamlit as st

from core import api_client as api
from core import styles
from core.state import cached_documents, mmss, safe

styles.kicker("STUDY MODE"); st.title("🧠 Interactive Learning")
# this page shows a document and a chat side by side, so it may use more of the screen
st.markdown("<style>.block-container{max-width:1500px !important;}</style>", unsafe_allow_html=True)

ICONS = {"pdf": "📕", "pptx": "📊", "docx": "📝", "video": "🎬"}
BOT_ICON = Path(__file__).resolve().parent.parent / "assets" / "chat-bot-icon.png"
BOT_AVATAR = str(BOT_ICON) if BOT_ICON.is_file() else "🤖"   # chat avatar of the study buddy
ss = st.session_state
ss.setdefault("buddy_open", False)
ss.setdefault("buddy_chats", {})   # "course|document" -> [messages]


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


# --- 1. pick a document ---------------------------------------------------------------------------
docs = safe(cached_documents)
if docs is None:
    st.stop()
if not docs:
    st.info("You have no indexed documents yet.")
    st.page_link("views/materials.py", label="Upload materials →")
    st.stop()

by_key = {f"{d['course']}|{d['document']}": d for d in docs}
key = st.selectbox(
    "Document", list(by_key), key="iv_doc",
    format_func=lambda k: f"{ICONS.get(by_key[k]['type'], '📄')} {by_key[k]['document']} · "
                          f"{by_key[k]['course']} · week {by_key[k]['lecture']}")
d = by_key[key]
info = safe(outline, d["course"], d["document"])
if info is None:
    st.stop()
kind, units = info["type"], info["units"]
nav = f"iv_pos_{key}"   # viewer position: PDF page number, or 1-based unit index otherwise
seek = f"iv_seek_{key}"  # video position in seconds
pdf_pages = info["pages"] if kind == "pdf" and info["has_file"] else None
last = pdf_pages or len(units)


def move(step: int):
    ss[nav] = min(max(1, ss.get(nav, 1) + step), last)


def go_to(src: dict):
    """Chat source button → show that page / slide / section / moment in the viewer."""
    if kind == "video":
        if src.get("start_ms") is not None:
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
        return {"time_ms": int(ss.get(seek, 0)) * 1000}
    pos = int(ss.get(nav, 1))
    if pdf_pages:
        return {"page": pos}
    u = units[min(pos, len(units)) - 1]
    return {"page": u["page"], "section": u["section"]}


# --- floating chat-bot button (fixed bottom right, see styles.py) ---------------------------------
icon = bot_icon_uri()
if icon:
    st.markdown(f"<style>.st-key-buddy_fab button, .st-key-buddy_fab_open button"
                f"{{background-image:url('{icon}') !important;}}</style>", unsafe_allow_html=True)
st.button("Study Buddy" if icon else "🤖", key="buddy_fab_open" if ss.buddy_open else "buddy_fab",
          on_click=toggle_buddy,
          help="Close the study buddy" if ss.buddy_open else "Ask the study buddy about this document")

viewer, panel = (st.columns([3, 2], gap="medium") if ss.buddy_open else (st.container(), None))


# --- 2. document viewer ---------------------------------------------------------------------------
def doc_page(title: str, text: str):
    st.markdown(f'<div class="doc-page"><div class="doc-page-title">{html.escape(title)}</div>'
                f'{html.escape(text)}</div>', unsafe_allow_html=True)


with viewer:
    st.caption(f"{info['chunks']} indexed chunks · course {info['course']} · week {info['lecture']}")
    if kind == "video":
        dur = max(1, int(max((u["end_ms"] or 0) for u in units) // 1000))
        ss.setdefault(seek, 0)
        if info["has_file"]:
            st.video(api.study_file_url(d["course"], d["document"]), start_time=int(ss[seek]))
        else:
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
            h1, h2, h3 = st.columns([3, 2, 1], vertical_alignment="center")
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
                        st.markdown(m["content"])
                        if m["role"] == "assistant":
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
                        "provider": ss.cfg.get("provider"), **current_location()}
                meta: dict = {}
                with box:
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.markdown(shown)
                    with st.chat_message("assistant", avatar=BOT_AVATAR):
                        t0 = perf_counter()
                        try:
                            answer = st.write_stream(api.study_chat_stream(body, meta))
                        except api.ApiError as exc:
                            st.error(f"Backend error: {exc}")
                            answer = None
                if answer:
                    total = perf_counter() - t0
                    chat.append({"role": "user", "content": shown})
                    chat.append({"role": "assistant", "content": answer if isinstance(answer, str)
                                 else "".join(map(str, answer)),
                                 "sources": meta.get("sources", []),
                                 "timing": f"⚡ search {meta.get('retrieval_ms', '–')} ms · "
                                           f"total {total:.1f} s"})
                    st.rerun()
