import html

import streamlit as st
from core.api_client import BASE_URL
from core.state import backend_health, init_state, is_private, log_out
from core import styles

st.set_page_config(
    page_title="Interactive Learning Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="auto",   # open on a computer, closed on a phone
)
init_state(); styles.inject()

# nothing but the login page until the student has logged in; then the pages in three groups
pages = [st.Page("views/login.py", title="Log in", icon=":material/login:")] if not st.session_state.auth else {
    "": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
    "Learn": [st.Page("views/qa.py", title="Q&A", icon=":material/forum:"),
              st.Page("views/interactive.py", title="Interactive Learning", icon=":material/menu_book:")],
    "Practice": [st.Page("views/question_gen.py", title="Question Generation", icon=":material/quiz:"),
                 st.Page("views/history.py", title="My Results", icon=":material/insights:")],
    "Manage": [st.Page("views/materials.py", title="Materials", icon=":material/folder_open:"),
               st.Page("views/settings.py", title="Settings", icon=":material/settings:"),
               st.Page("views/survey.py", title="Feedback", icon=":material/rate_review:")],
}
with st.sidebar:
    st.markdown('<div class="side-brand"><div class="logo">L</div><div><div class="name">Learning Assistant</div>'
                '<div class="sub">Hybrid RAG · Privacy-focused</div></div></div>', unsafe_allow_html=True)
    if st.session_state.auth:
        a, b = st.columns([3, 1], vertical_alignment="center")
        no = html.escape(st.session_state.auth["student_no"])
        a.markdown(f'<div class="side-user"><span class="av">{no[:2].upper()}</span>{no}</div>',
                   unsafe_allow_html=True)
        if b.button("Log out", icon=":material/logout:", help="Log out", key="logout"):
            log_out()
    if is_private():
        st.success("Privacy mode: no data leaves your environment", icon=":material/lock:")
    else:
        st.info("Hybrid mode: only 3–5 context chunks are sent to the API", icon=":material/shield:")
    h = backend_health()
    if h:
        st.caption(f"🟢 Backend connected · default LLM: {h['provider']}")
        if h.get("models") == "loading":
            st.warning("⏳ Loading the embedding and reranker models (the first start downloads ~4 GB). "
                       "Questions will be slow until this finishes.")
        elif h.get("models") == "failed":
            st.error(f"Model loading failed: {h.get('models_error')}")
    else:
        st.error(f"🔴 Backend offline ({BASE_URL})")
        st.caption("Start it with `uvicorn backend.main:app`")
st.navigation(pages).run()