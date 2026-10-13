import pandas as pd, streamlit as st
from core import styles
from core import api_client as api
from core.state import backend_config, cached_courses, cached_documents, invalidate, mmss, safe
styles.kicker("RAG PIPELINE"); st.title("📚 Materials")
cfg = st.session_state.cfg
NEW = "➕ New course…"

with st.container(border=True):
    known = [c["course"] for c in (safe(cached_courses) or [])]
    c1, c2 = st.columns(2)
    pick = c1.selectbox("Course", known + [NEW])
    course = (c1.text_input("New course code", placeholder="e.g. CENG336").replace(" ", "").upper()
              if pick == NEW else pick)
    lecture = c2.number_input("Week / lecture number", 1, 30, 1)
    files = st.file_uploader("Course materials", type=["pdf", "pptx", "docx", "mp4", "mkv", "avi"], accept_multiple_files=True)
    ok = st.checkbox("I confirm that these materials are approved by the instructor and the institution has the right to use them (Turkish Law No. 5846).")
    b = safe(backend_config) or {}
    if b:
        st.caption(f"Chunking: {b['chunk_size']} tokens / {b['chunk_overlap']} overlap · Embedding: {b['embedding_model']} (local) · Video STT: faster-whisper ({b['whisper_model']})")
    if st.button("Index materials", type="primary", disabled=not (files and ok and course)):
        with st.status("Processing…", expanded=True) as s:
            bar = st.progress(0.0)
            failed = 0
            for i, f in enumerate(files):
                st.write(f"• {f.name}: extraction → chunking → embedding → ChromaDB")
                try:
                    r = api.ingest_file(f.name, f.getvalue(), course, int(lecture), f.type or "application/octet-stream")
                    st.write(f"  ✓ {r['chunks']} chunks")
                except api.ApiError as exc:
                    failed += 1
                    st.write(f"  ✗ {exc}")
                bar.progress((i + 1) / len(files))
            invalidate()
            s.update(label="Indexing complete" if not failed else f"Indexing finished with {failed} error(s)",
                     state="complete" if not failed else "error")

st.subheader("Indexed documents")
docs = safe(cached_documents)
if docs:
    df = pd.DataFrame(docs).rename(columns={"document": "Document", "course": "Course", "lecture": "Week",
                                            "type": "Type", "chunks": "Chunks"})
    st.dataframe(df, width="stretch", hide_index=True)

    low = safe(api.low_confidence, cfg["conf_threshold"]) or []
    if low:
        st.warning(f"The STT confidence is below {cfg['conf_threshold']} for {len(low)} video segments. Reprocess with a larger model or correct them manually.")
        with st.expander("✏️ Edit transcript (instructor)"):
            seg = st.selectbox("Segment", low, format_func=lambda x: f"{x['document']} · {mmss(x['start_ms'])}–{mmss(x['end_ms'])} · confidence {x['confidence']:.2f}")
            text = st.text_area("Corrected text", seg["text"], height=100, key=f"fix_{seg['id']}")
            if st.button("Save correction", disabled=not text.strip()):
                if safe(api.correct_chunk, seg["id"], text):
                    st.success("Saved and re-embedded."); st.rerun()

    with st.expander("🗑️ Remove materials"):
        courses = {c["course"]: c for c in (safe(cached_courses) or [])}
        d1, d2 = st.columns(2)
        dc = d1.selectbox("Course to remove", list(courses), key="del_course")
        dl = d2.selectbox("Lecture", [None] + (courses[dc]["lectures"] if dc else []), key="del_lecture",
                          format_func=lambda x: "All lectures" if x is None else f"Lecture {x}")
        if st.button("Remove selected", disabled=not dc):
            n = safe(api.delete_course, dc, dl)
            if n is not None:
                invalidate(); st.success(f"Removed {n} chunks."); st.rerun()
        st.divider()
        sure = st.checkbox("I understand that this deletes every indexed chunk.")
        if st.button("Clear entire index", disabled=not sure):
            if safe(api.clear_index):
                invalidate(); st.rerun()
elif docs is not None:
    st.info("No documents yet.")
