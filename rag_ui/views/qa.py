import streamlit as st
from core import backend as be, styles
from core.state import label_for, mmss, is_private
styles.kicker("CONTEXT-GROUNDED Q&A"); st.title("💬 Q&A")

c1, c2, c3 = st.columns([2, 1, 2])
course = c1.selectbox("Course", list(be.COURSES), key="qa_course")
lecture = c2.selectbox("Lecture number", range(1, 15), index=4)
watching = c3.toggle("🎬 I'm watching the lecture video")
ts = st.slider("Watched time range (min)", 0, 90, (12, 14)) if watching else None
if ts: st.caption(f"Retrieval will prioritize {ts[0]}:00–{ts[1]}:00.")

def render(m):
    st.markdown(m["answer"])
    cols = st.columns(4)
    cols[0].markdown(label_for(m["faithfulness"]))
    cols[1].caption(f"Faithfulness: {m['faithfulness']}")
    cols[2].caption(f"Retrieval: {m['retrieval_ms']} ms · Total: {m['total_s']} s")
    cols[3].caption(f"Sent to API: ~{m['tokens_sent']} tokens")
    with st.expander(f"📎 Sources ({len(m['sources'])})"):
        for i, s in enumerate(m["sources"], 1):
            loc = f"{mmss(s['start_ms'])}–{mmss(s['end_ms'])}" if s["type"] == "video" else f"page/slide {s['page']}"
            st.markdown(f"**[{i}]** `{s['type']}` {s['document']} · {loc} · similarity {s['score']}")
            st.caption(s["text"])
    with st.expander("🛡️ Preview of context sent externally"):
        if is_private(): st.success("Privacy mode: no data was sent externally (Ollama).")
        else: st.warning(f"Only {len(m['sources'])} chunks were sent — the raw documents are never sent.")
        st.caption(f"Filtering: {m['filtered_from']} chunks → {m['candidates']} candidates → {len(m['sources'])} context chunks")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        if m["role"] == "user": st.markdown(m["content"])
        else: render(m)
if q := st.chat_input("Ask a question about the course…"):
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Filtering → retrieving → reranking → generating response…"):
            r = be.ask(q, course, lecture, ts, st.session_state.cfg)
        render(r)
    st.session_state.chat.append({"role": "assistant", "content": r["answer"], **r})
if st.session_state.chat and st.sidebar.button("Clear chat"):
    st.session_state.chat = []; st.rerun()