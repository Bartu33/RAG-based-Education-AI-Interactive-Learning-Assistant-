import streamlit as st
from core import styles
from core.state import DEFAULT_CFG, PROVIDERS
styles.kicker("CONFIGURATION"); st.title("⚙️ Settings")
c = st.session_state.cfg
st.subheader("LLM provider")
c["provider"] = st.radio("Provider (LLM_PROVIDER)", PROVIDERS, index=PROVIDERS.index(c["provider"]))
st.subheader("Chunking and retrieval")
a, b = st.columns(2)
c["chunk_size"] = a.slider("Chunk size (tokens)", 200, 800, c["chunk_size"], 50)
c["overlap"] = b.slider("Overlap (tokens)", 0, 150, c["overlap"], 10)
c["top_k_candidates"] = a.slider("Candidate chunks (similarity search)", 5, 30, c["top_k_candidates"])
c["top_k_context"] = b.slider("Context chunks (after reranking)", 1, 8, c["top_k_context"])
st.subheader("Models and thresholds")
embs = ["BAAI/bge-m3", "multilingual-e5-large"]
c["embedding"] = st.selectbox("Embedding model", embs, index=embs.index(c["embedding"]))
c["conf_threshold"] = st.slider("STT confidence threshold", 0.0, 1.0, c["conf_threshold"], 0.05)
t1, t2 = st.columns(2)
c["th_grounded"] = t1.number_input("Grounded threshold", 0.0, 1.0, c["th_grounded"], 0.05)
c["th_partial"] = t2.number_input("Partially grounded threshold", 0.0, 1.0, c["th_partial"], 0.05)
if st.button("Reset to defaults"):
    st.session_state.cfg = dict(DEFAULT_CFG); st.rerun()