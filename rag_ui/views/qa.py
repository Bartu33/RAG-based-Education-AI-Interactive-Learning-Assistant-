import uuid
import streamlit as st
from core import styles
from core import api_client as api
from core.state import course_picker, feedback_widget, label_for, mmss
styles.kicker("CONTEXT-GROUNDED Q&A"); st.title("💬 Q&A")

course, lecture = course_picker("qa")
watching = st.toggle("🎬 I'm watching the lecture video")
ts = st.slider("Watched time range (min)", 0, 90, (12, 14)) if watching else None
if ts: st.caption(f"Retrieval will prioritize {ts[0]}:00–{ts[1]}:00.")

def render(m):
    st.markdown(m["answer"])
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

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        if m["role"] == "user": st.markdown(m["content"])
        else: render(m)
if q := st.chat_input("Ask a question about the course…"):
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Filtering → retrieving → reranking → generating response…"):
                r = api.ask(q, course, lecture, ts, st.session_state.cfg)
        except api.ApiError as exc:
            st.error(f"Backend error: {exc}")
        else:
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": r["answer"], "mid": uuid.uuid4().hex[:12],
                                          "question": q, "course": course, **r})
            st.rerun()  # drawn by the loop above, so its feedback buttons keep a stable key
if st.session_state.chat and st.sidebar.button("Clear chat"):
    st.session_state.chat = []; st.rerun()
