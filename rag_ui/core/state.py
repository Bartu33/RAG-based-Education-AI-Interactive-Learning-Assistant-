import re
from typing import Optional

import streamlit as st

from core import api_client as api
from core.api_client import ApiError

PROVIDERS = ["Groq (GPT-OSS 120B)", "OpenAI (GPT-4o-mini)", "Ollama (Mistral) — Gizlilik modu"]
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
    st.session_state.setdefault("checked", {})   # question id -> {"correct": bool} (choice questions)
    st.session_state.setdefault("graded", {})    # question id -> rubric result (open-ended questions)
    st.session_state.setdefault("auth", None)     # {"token", "student_no"} after logging in


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


def feedback_widget(kind: str, rating_id: str, context: dict) -> None:
    """👍/👎 under an answer or an evaluation; 👎 opens an optional comment. Sent to the backend at
    once; a changed thumb or an added comment replaces the earlier rating (same id)."""
    key = f"fb_{rating_id}"
    sent = st.session_state.setdefault("ratings_sent", {})
    value = st.feedback("thumbs", key=key)
    if value is None:
        return
    up = value == 1
    if sent.get(rating_id) != up:
        if safe(api.rate, rating_id, kind, up, "", context) is not None:
            sent[rating_id] = up
    if up:
        st.caption("Thanks for the feedback!")
        return
    with st.form(f"why_{rating_id}", border=False):
        why = st.text_input("What was wrong? (optional)", key=f"why_txt_{rating_id}",
                            placeholder="e.g. wrong source, incomplete, too long…")
        if st.form_submit_button("Send") and why.strip():
            if safe(api.rate, rating_id, kind, False, why, context) is not None:
                st.caption("Thanks, noted.")


def render_evaluation(r: dict, rating_id: Optional[str] = None, question: str = "") -> None:
    """Rubric score, per-dimension bars, feedback and follow-up question of one graded answer.
    With rating_id, a 👍/👎 asks whether the feedback was helpful."""
    c1, c2 = st.columns([1, 2])
    c1.metric("Overall score", f"{r['total']} / 100")
    with c2:
        for d in r["dims"]:
            st.write(f"**{d['name']}** · weight {int(d['weight']*100)}% · {int(d['score']*100)}/100")
            st.progress(d["score"])
    st.markdown("**Detailed feedback**")
    for f in r["feedback"]: st.write(f)
    st.caption(f"📎 Reference: {r['source']}")
    if r["follow_up_question"]:
        st.info(f"🎯 **Follow-up practice:** {r['follow_up_question']}")
    if rating_id:
        st.caption("Was this feedback helpful?")
        feedback_widget("evaluation", rating_id, {"question": question, "score": r["total"]})


def token() -> str:
    return st.session_state.auth["token"]


def log_out(reason: Optional[str] = None) -> None:
    """Forget the session (the backend token is dropped too) and show the login page."""
    if st.session_state.auth:
        try:
            api.logout(token())
        except ApiError:
            pass
    for key in list(st.session_state):  # the next person on this browser starts clean
        if key != "cfg":
            del st.session_state[key]
    init_state()
    st.session_state.login_notice = reason
    st.rerun()


def record_attempt(q: dict, student_answer: str, is_correct=None, result: Optional[dict] = None) -> None:
    """Save one checked / graded answer to the student's history (My Results). A failure is shown
    but does not block practising; an expired login sends the student back to the login page."""
    if not q.get("test_id"):
        st.warning("This answer could not be saved to My Results (the test was not registered).")
        return
    attempt = {"test_id": q["test_id"], "question_id": q["id"], "student_answer": student_answer,
               "is_correct": is_correct}
    if result:
        attempt.update(score=result["total"], feedback=result["feedback"], dims=result["dims"])
    try:
        api.save_attempt(token(), attempt)
    except ApiError as exc:
        if exc.status == 401:
            log_out("Your session expired, please log in again.")
        st.error(f"Could not save to My Results: {exc}")


# same rule as backend/llm/prompts.py (the UI does not import backend code): a streamed answer is
# split here, a non-streamed one arrives already split in "beyond"
BEYOND_MARKER = "[BEYOND]"
_BEYOND_RE = re.compile(r"\**\s*\[BEYOND\]\s*\**\s*:?", re.IGNORECASE)


def split_beyond(text: str) -> tuple[str, Optional[str]]:
    m = _BEYOND_RE.search(text)
    if not m:
        return text.strip(), None
    return text[:m.start()].strip(), _BEYOND_RE.sub("", text[m.end():]).strip() or None


class BeyondStream:
    """Wraps a streamed answer for st.write_stream: while writing, the [BEYOND] marker is shown as
    a separator line; the raw text is kept in .raw so it can be split afterwards."""
    SEPARATOR = "\n\n---\n💡 **Beyond the course material** (general knowledge)\n\n"

    def __init__(self, pieces):
        self.pieces, self.raw = pieces, ""

    def __iter__(self):
        buf = ""
        for piece in self.pieces:
            self.raw += piece
            buf = _BEYOND_RE.sub(self.SEPARATOR, buf + piece)
            hold = next((n for n in range(len(BEYOND_MARKER) - 1, 0, -1)
                         if buf.upper().endswith(BEYOND_MARKER[:n])), 0)  # marker may be split
            if len(buf) > hold:
                yield buf[:len(buf) - hold]
                buf = buf[len(buf) - hold:]
        if buf:
            yield buf


def is_choice(q: dict) -> bool:
    """Multiple choice and true/false questions are checked on the spot, not graded with the rubric."""
    return bool(q.get("options")) or "true" in q["type"].lower()


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
