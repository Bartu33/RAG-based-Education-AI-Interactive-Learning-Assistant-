import streamlit as st
from core import styles
from core.state import PROVIDERS, backend_defaults, backend_config, safe
styles.kicker("CONFIGURATION"); st.title("⚙️ Settings")
c = st.session_state.cfg
st.subheader("LLM provider")
c["provider"] = st.radio("Provider (LLM_PROVIDER)", PROVIDERS, index=PROVIDERS.index(c["provider"]))
st.subheader("Retrieval")
a, b = st.columns(2)
c["top_k_candidates"] = a.slider("Candidate chunks (similarity search)", 5, 30, c["top_k_candidates"])
c["top_k_context"] = b.slider("Context chunks (after reranking)", 1, 8, c["top_k_context"])
st.subheader("Thresholds")
c["conf_threshold"] = st.slider("STT confidence threshold (flags transcript segments to correct)", 0.0, 1.0, c["conf_threshold"], 0.05)
t1, t2 = st.columns(2)
c["th_grounded"] = t1.number_input("Grounded threshold", 0.0, 1.0, c["th_grounded"], 0.05)
c["th_partial"] = t2.number_input("Partially grounded threshold", 0.0, 1.0, c["th_partial"], 0.05)
st.subheader("Indexing (config.yaml)")
bk = safe(backend_config)
if bk:
    st.caption("These are fixed when materials are indexed. Change them in config.yaml and re-index; "
               "the index refuses to mix embedding models.")
    st.table({"Embedding model": [bk["embedding_model"]], "Reranker": [bk["reranker_model"]],
              "Chunk size (tokens)": [bk["chunk_size"]], "Overlap (tokens)": [bk["chunk_overlap"]],
              "Video STT": [f"faster-whisper ({bk['whisper_model']})"]})
if st.button("Reset to defaults"):
    st.session_state.cfg = backend_defaults(); st.rerun()
