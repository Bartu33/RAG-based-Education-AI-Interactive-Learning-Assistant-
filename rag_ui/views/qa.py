import uuid
import streamlit as st
from core import styles
from core import api_client as api
from core import board
from core.state import course_picker, feedback_widget, label_for, mmss
styles.kicker("CONTEXT-GROUNDED Q&A"); st.title("💬 Q&A")

# This page uses the full width: the content starts where the question box starts (right of
# Clear chat) and ends left of the lecture video that plays in the bottom-right corner
# (core/board.py: 340 px wide, 24 px from the edge), so neither of them covers an answer.
QA_LAYOUT = """
<style>
:root{ --qa-side:1.5rem; --qa-clear:170px; --qa-gap:1rem; --qa-video:calc(340px + 24px + 1rem); }
@media (max-width: 1100px){ :root{ --qa-clear:120px; --qa-video:1.5rem; } }
.block-container{ max-width:none !important; padding-right:var(--qa-video) !important;
  padding-left:calc(var(--qa-side) + var(--qa-clear) + var(--qa-gap)) !important; }
[data-testid="stBottomBlockContainer"]{ max-width:none !important; padding-left:var(--qa-side) !important;
  padding-right:var(--qa-video) !important; }
.st-key-qa_bottom [data-testid="stHorizontalBlock"]{ gap:var(--qa-gap) !important; }
.st-key-qa_bottom [data-testid="stColumn"]:first-child{ flex:0 0 var(--qa-clear) !important;
  width:var(--qa-clear) !important; min-width:0 !important; }
.st-key-qa_bottom [data-testid="stColumn"]:last-child{ flex:1 1 auto !important; width:auto !important; }
</style>
"""
st.markdown(QA_LAYOUT, unsafe_allow_html=True)

def clear_chat():
    st.session_state.chat = []
    st.session_state.board_focus = None   # also stops the video playing in the corner


course, lecture = course_picker("qa")
t1, t2, t3 = st.columns(3)
watching = t1.toggle("🎬 I'm watching the lecture video")
use_board = t2.toggle("🧑‍🏫 Teaching board", value=True,
                      help="Video answers play in the corner; the matching slide / page is shown with the "
                           "relevant lines highlighted; math questions are solved step by step on a whiteboard.")
make_video = t3.toggle("🎞️ Explanation video", value=True, disabled=not use_board,
                       help="Also record the explanation as a video: the lecturer's voice (Piper, local), "
                            "the slides / whiteboard and the lecture recording in the corner.")
ts = st.slider("Watched time range (min)", 0, 90, (12, 14)) if watching else None
if ts: st.caption(f"Retrieval will prioritize {ts[0]}:00–{ts[1]}:00.")

def render(m):
    st.markdown(board.fix_markdown(m["answer"]))
    found = m.get("found", True)
    cols = st.columns(4)
    if found:
        cols[0].markdown(label_for(m["faithfulness"]))
        cols[1].caption(f"Faithfulness: {'—' if m['faithfulness'] is None else m['faithfulness']}")
    else:
        cols[0].markdown("ℹ️ Not found in course materials")
        cols[1].caption("Nothing to verify")
    cols[2].caption(f"Retrieval: {m['retrieval_ms']} ms · Total: {m['total_s']} s")
    cols[3].caption(f"Sent to API: ~{m['tokens_sent']} tokens")
    if m.get("beyond"):  # after the label: the grounding score covers only the part above
        styles.beyond_box(m["beyond"], m["mid"])
    if found:  # the lecturer explains the course-material part (after the answer, labels and beyond box)
        board.render(m, m["course"])
    title = f"📎 Sources ({len(m['sources'])})" if found else f"🔎 Closest passages checked ({len(m['sources'])})"
    if m["sources"]:
        with st.expander(title):
            for i, s in enumerate(m["sources"], 1):
                if s["type"] == "video" and s["start_ms"] is not None:
                    loc = f" · {mmss(s['start_ms'])}–{mmss(s['end_ms'])}"
                elif s["page"]:
                    loc = f" · {'slide' if s['type'] == 'pptx' else 'page'} {s['page']}"
                else:
                    loc = ""
                st.markdown(f"**[{i}]** `{s['type']}` {s['document']}{loc} · score {s['score']}")
                st.caption(s["text"])
    with st.expander("🛡️ Preview of context sent externally"):
        if m.get("provider") == "ollama": st.success("Privacy mode: no data was sent externally (Ollama).")
        elif not m["sources"]: st.info("Nothing was sent: no relevant course material was found.")
        else: st.warning(f"Only {len(m['sources'])} chunks were sent to {m.get('provider') or 'the LLM'} — the raw documents are never sent.")
        st.caption(f"Filtering: {m['filtered_from']} chunks → {m['candidates']} candidates → {len(m['sources'])} context chunks")
    feedback_widget("qa", m["mid"], {"question": m["question"], "course": m["course"], "found": found,
                                     "faithfulness": m["faithfulness"], "provider": m.get("provider")})

board.corner(st.session_state.chat)   # first, so the video starts while the lecturer explains
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        if m["role"] == "user": st.markdown(m["content"])
        else: render(m)
with st.bottom, st.container(key="qa_bottom"):   # pinned to the bottom: Clear chat left of the question box
    b1, b2 = st.columns([1, 6], vertical_alignment="center")
    b1.button("🗑️ Clear chat", on_click=clear_chat, disabled=not st.session_state.chat,
              width="stretch", help="Delete all questions and answers of this chat")
    q = b2.chat_input("Ask a question about the course…")
if q:
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
