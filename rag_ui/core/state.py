import streamlit as st

PROVIDERS = ["Groq (Llama 3)", "OpenAI (GPT-4o-mini)", "Ollama (Mistral) — Gizlilik modu"]

DEFAULT_CFG = {
    "provider": PROVIDERS[0], "embedding": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3", "chunk_size": 500, "overlap": 75,
    "top_k_candidates": 20, "top_k_context": 4, "conf_threshold": 0.60,
    "th_grounded": 0.85, "th_partial": 0.50, "whisper": "faster-whisper (large-v3)",
}

def init_state():
    st.session_state.setdefault("cfg", dict(DEFAULT_CFG))
    st.session_state.setdefault("docs", [])
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("eval_target", None)
    st.session_state.setdefault("eval_history", [])

def label_for(score: float):
    c = st.session_state.cfg
    if score >= c["th_grounded"]: return "🟢 Grounded"
    if score >= c["th_partial"]: return "🟡 Partially grounded"
    return "🔴 Unsupported"

def mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"

def is_private() -> bool:
    return "Ollama" in st.session_state.cfg["provider"]