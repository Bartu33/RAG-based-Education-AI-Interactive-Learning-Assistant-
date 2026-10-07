import pandas as pd, streamlit as st
from core import backend as be, styles
styles.kicker("RAG PIPELINE"); st.title("📚 Materials")
cfg = st.session_state.cfg
with st.container(border=True):
    c1, c2 = st.columns(2)
    course = c1.selectbox("Course", list(be.COURSES), format_func=lambda k: f"{k} · {be.COURSES[k]}")
    lecture = c2.number_input("Week / lecture number", 1, 14, 1)
    files = st.file_uploader("Course materials", type=["pdf", "pptx", "docx", "mp4", "mkv", "avi"], accept_multiple_files=True)
    ok = st.checkbox("I confirm that these materials are approved by the instructor and the institution has the right to use them (Turkish Law No. 5846).")
    st.caption(f"Chunking: {cfg['chunk_size']} tokens / {cfg['overlap']} overlap · Embedding: {cfg['embedding']} (local) · Video STT: {cfg['whisper']}")
    if st.button("Index materials", type="primary", disabled=not (files and ok)):
        with st.status("Processing…", expanded=True) as s:
            bar = st.progress(0.0)
            for stage, pct, recs in be.ingest(files, course, lecture):
                st.write(f"• {stage}"); bar.progress(pct)
                if recs: st.session_state.docs += recs
            s.update(label="Indexing complete", state="complete")
st.subheader("Indexed documents")
if st.session_state.docs:
    df = pd.DataFrame(st.session_state.docs)
    st.dataframe(df, use_container_width=True, hide_index=True)
    low = int(df["Low-confidence segments"].sum())
    if low:
        st.warning(f"The STT confidence is below {cfg['conf_threshold']} for {low} video segments. Reprocess with a larger model or correct them manually.")
        with st.expander("✏️ Edit transcript (instructor)"):
            st.text_area("Low-confidence segment", "the nyquist rate of this signal …", height=80)
            st.button("Save correction")
    if st.button("Clear entire index"):
        st.session_state.docs = []; st.rerun()
else:
    st.info("No documents yet.")