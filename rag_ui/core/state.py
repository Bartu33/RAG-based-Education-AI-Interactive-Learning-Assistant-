from typing import Optional

import streamlit as st

from core import api_client as api
from core.api_client import ApiError

PROVIDERS = ["Groq (Llama 3)", "OpenAI (GPT-4o-mini)", "Ollama (Mistral) — Gizlilik modu"]
QTYPES = ["Multiple Choice", "True/False", "Short Answer", "Conceptual/Computational"]

DEFAULT_CFG = {
    "provider": PROVIDERS[0], "top_k_candidates": 20, "top_k_context": 4,
    "conf_threshold": 0.60, "th_grounded": 0.85, "th_partial": 0.50,
}


def backend_defaults() -> dict:
    """DEFAULT_CFG with the values config.yaml holds (when the backend is reachable)."""
    cfg = dict(DEFAULT_CFG)
    try:
        b = api.public_config()
    except ApiError:
        return cfg
    provider = next((p for p in PROVIDERS if b["provider"] in p.lower()), cfg["provider"])
    cfg.update(provider=provider, top_k_candidates=b["top_candidates"],
               top_k_context=b["top_context"], conf_threshold=b["confidence_threshold"],
               th_grounded=b["grounded_threshold"], th_partial=b["partial_threshold"])
    return cfg


def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = backend_defaults()
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("eval_target", None)
    st.session_state.setdefault("eval_history", [])


# --- cached backend reads (cleared after every write) ---
@st.cache_data(ttl=10, show_spinner=False)
def backend_health() -> Optional[dict]:
    try:
        return api.health()
    except ApiError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def backend_config() -> dict:
    return api.public_config()


@st.cache_data(ttl=30, show_spinner=False)
def cached_courses() -> list[dict]:
    return api.courses()


@st.cache_data(ttl=30, show_spinner=False)
def cached_documents() -> list[dict]:
    return api.documents()


def invalidate():
    cached_courses.clear()
    cached_documents.clear()


def safe(fn, *args, **kwargs):
    """Call the backend; show the error and return None instead of crashing the page."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        st.error(f"Backend error: {exc}")
        return None


def course_picker(key: str, all_lectures: bool = True) -> tuple[str, Optional[int]]:
    """Course + lecture selectors filled from the index. Stops the page if nothing is indexed."""
    courses = safe(cached_courses)
    if courses is None:
        st.stop()
    if not courses:
        st.info("No course material is indexed yet.")
        st.page_link("views/materials.py", label="Upload materials →")
        st.stop()
    by_name = {c["course"]: c for c in courses}
    c1, c2 = st.columns(2)
    course = c1.selectbox("Course", list(by_name), key=f"{key}_course",
                          format_func=lambda k: f"{k} · {by_name[k]['documents']} documents")
    options = ([None] if all_lectures else []) + by_name[course]["lectures"]
    lecture = c2.selectbox("Lecture number", options, key=f"{key}_lecture",
                           format_func=lambda x: "All lectures" if x is None else f"Lecture {x}")
    return course, lecture


def label_for(score: Optional[float]):
    if score is None:
        return "⚪ Not scored"
    c = st.session_state.cfg
    if score >= c["th_grounded"]: return "🟢 Grounded"
    if score >= c["th_partial"]: return "🟡 Partially grounded"
    return "🔴 Unsupported"


def mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def is_private() -> bool:
    return "Ollama" in st.session_state.cfg["provider"]
