import uuid
import streamlit as st
from core import styles
from core.ui_components import empty_state, page_header, source_chips_html, source_card_html
from core import api_client as api
from core import board
from core.state import course_picker, feedback_widget, label_for, mmss
page_header("Q&A", "Context-grounded answers", "Ask about your course. Every answer shows the page, slide or minute it comes from.", "forum")

# This page uses the full width: the content starts at the same left edge as on every other page
# (the question box too, with the Clear chat icon button on its right) and ends left of the lecture
# video that plays in the bottom-right corner (core/board.py: 340 px wide, 24 px from the edge), so
# the video never covers an answer.
QA_LAYOUT = """
<style>
:root{ --qa-side:clamp(1.25rem, 3vw, 3.5rem); --qa-clear:56px; --qa-gap:.75rem; --qa-video:calc(340px + 24px + 1rem); }
@media (max-width: 1100px){ :root{ --qa-video:1.5rem; } }
.block-container{ max-width:none !important; padding-right:var(--qa-video) !important;
  padding-left:var(--qa-side) !important; }
[data-testid="stBottomBlockContainer"]{ max-width:none !important; padding-left:var(--qa-side) !important;
  padding-right:var(--qa-video) !important; }
.st-key-qa_bottom [data-testid="stHorizontalBlock"]{ gap:var(--qa-gap) !important; flex-wrap:nowrap !important; }
.st-key-qa_bottom [data-testid="stColumn"]:first-child{ flex:1 1 auto !important; width:auto !important; min-width:0 !important; }
.st-key-qa_bottom [data-testid="stColumn"]:last-child{ flex:0 0 var(--qa-clear) !important;
  width:var(--qa-clear) !important; min-width:0 !important; }
/* Clear chat: icon only (the label stays for screen readers and the tooltip) */
.st-key-qa_bottom [data-testid="stColumn"]:last-child button{ width:var(--qa-clear) !important; height:56px;
  min-height:0; padding:0 !important; border-radius:var(--r-lg) !important; box-shadow:var(--shadow-2) !important; }
.st-key-qa_bottom [data-testid="stColumn"]:last-child button [data-testid="stIconMaterial"]{ font-size:1.3rem; }
.st-key-qa_bottom [data-testid="stColumn"]:last-child button p{ position:absolute; width:1px; height:1px; overflow:hidden;
  clip:rect(0 0 0 0); white-space:nowrap; }
.st-key-qa_bottom [data-testid="stColumn"]:last-child button:not(:disabled):hover{ border-color:var(--danger-line) !important;
  color:var(--danger) !important; background-color:var(--danger-soft) !important; }
</style>
"""
if not any(m.get("mid") == st.session_state.get("board_focus") and (m.get("board") or {}).get("video")
           for m in st.session_state.chat):
    QA_LAYOUT = QA_LAYOUT.replace("--qa-video:calc(340px + 24px + 1rem);", "--qa-video:1.5rem;")   # no corner video now
st.markdown(QA_LAYOUT, unsafe_allow_html=True)

def clear_chat():
    st.session_state.chat = []
    st.session_state.board_focus = None   # also stops the video playing in the corner


course, lecture = course_picker("qa")
t1, t2, t3 = st.columns(3)
watching = t1.toggle(":material/smart_display: I'm watching the lecture video")
use_board = t2.toggle(":material/co_present: Teaching board", value=True,
                      help="Video answers play in the corner; the matching slide / page is shown with the "
                           "relevant lines highlighted; math questions are solved step by step on a whiteboard.")
make_video = t3.toggle(":material/movie: Explanation video", value=True, disabled=not use_board,
                       help="Also record the explanation as a video: the lecturer's voice (Piper, local), "
                            "the slides / whiteboard and the lecture recording in the corner.")
ts = st.slider("Watched time range (min)", 0, 90, (12, 14)) if watching else None
if ts: st.caption(f"Retrieval will prioritize {ts[0]}:00–{ts[1]}:00.")

def render(m):
    st.markdown(board.fix_markdown(m["answer"]))
    found = m.get("found", True)
    if found and m["sources"]:   # where the answer comes from, visible without opening anything
        st.markdown(source_chips_html(m["sources"]), unsafe_allow_html=True)
    # one quiet line: grounding label + the numbers (same values as before)
    if found:
        badge = label_for(m["faithfulness"])
        score = f"Faithfulness {'—' if m['faithfulness'] is None else m['faithfulness']}"
    else:
        badge, score = ":gray-badge[:material/info: Not found in course materials]", "Nothing to verify"
    with st.container(key=f"meta_{m['mid']}"):
        st.markdown(f"{badge} :gray[{score} · Retrieval {m['retrieval_ms']} ms · Total {m['total_s']} s · "
                    f"Sent to API ~{m['tokens_sent']} tokens]")
    if m.get("beyond"):  # after the label: the grounding score covers only the part above
        styles.beyond_box(m["beyond"], m["mid"])
    if found:  # the lecturer explains the course-material part (after the answer, labels and beyond box)
        board.render(m, m["course"])
    title = f"Sources ({len(m['sources'])})" if found else f"Closest passages checked ({len(m['sources'])})"
    if m["sources"]:
        with st.expander(title, icon=":material/attach_file:" if found else ":material/search:"):
            for i, s in enumerate(m["sources"], 1):
                st.markdown(source_card_html(i, s, top=found and i == 1), unsafe_allow_html=True)
    with st.expander("Preview of context sent externally", icon=":material/shield:"):
        if m.get("provider") == "ollama": st.success("Privacy mode: no data was sent externally (Ollama).")
        elif not m["sources"]: st.info("Nothing was sent: no relevant course material was found.")
        else: st.warning(f"Only {len(m['sources'])} chunks were sent to {m.get('provider') or 'the LLM'} — the raw documents are never sent.")
        st.caption(f"Filtering: {m['filtered_from']} chunks → {m['candidates']} candidates → {len(m['sources'])} context chunks")
    feedback_widget("qa", m["mid"], {"question": m["question"], "course": m["course"], "found": found,
                                     "faithfulness": m["faithfulness"], "provider": m.get("provider")})

board.corner(st.session_state.chat)   # first, so the video starts while the lecturer explains
welcome = st.empty()   # shown only before the first question; cleared as soon as one is asked
if not st.session_state.chat:
    with welcome.container():
        empty_state("Ask your course",
                    "Answers come only from the approved material of the course you picked above. "
                    "Each one shows the page, slide or video minute it is based on, and how well it is "
                    "grounded in it.", logo=True)
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        if m["role"] == "user": st.markdown(m["content"])
        else: render(m)
with st.bottom, st.container(key="qa_bottom"):   # pinned to the bottom: the question box, Clear chat on its right
    b2, b1 = st.columns([6, 1], vertical_alignment="center")
    b1.button("Clear chat", on_click=clear_chat, disabled=not st.session_state.chat, icon=":material/delete_sweep:",
              width="stretch", help="Delete all questions and answers of this chat")
    q = b2.chat_input("Ask a question about the course…")
if q:
    welcome.empty()
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Filtering → retrieving → reranking → generating response…"):
                r = api.ask(q, course, lecture, ts, st.session_state.cfg)
            if use_board and r.get("found", True) and r["sources"]:
                with st.spinner("Preparing the teaching board…"):
                    r["board"] = board.fetch(q, r["answer"], course, lecture, r["sources"],
                                             st.session_state.cfg["provider"])
                if make_video and r["board"].get("view") in ("page", "whiteboard"):
                    r["video_job"] = board.start_video(q, course, r["board"])
        except api.ApiError as exc:
            st.error(f"Backend error: {exc}")
        else:
            st.session_state.chat.append({"role": "user", "content": q})
            mid = uuid.uuid4().hex[:12]
            st.session_state.chat.append({"role": "assistant", "content": r["answer"], "mid": mid,
                                          "question": q, "course": course, **r})
            if (r.get("board") or {}).get("video"):
                st.session_state.board_focus = mid   # the newest video answer plays in the corner
            st.rerun()  # drawn by the loop above, so its feedback buttons keep a stable key
