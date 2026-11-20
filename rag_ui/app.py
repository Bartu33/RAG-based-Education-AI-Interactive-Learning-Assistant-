import html

import streamlit as st
from core.api_client import BASE_URL
from core.state import backend_health, init_state, is_private, log_out
from core import styles
from core.ui_components import account_html, page_icon, sidebar_brand, sidebar_logo, status_html

st.set_page_config(
    page_title="Interactive Learning Assistant",
    page_icon=page_icon(),          # the app logo (assets/favicon.png)
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
if st.session_state.auth:
    sidebar_logo()    # small logo top-left while the sidebar is collapsed
with st.sidebar:
    sidebar_brand()   # logo + name, top-left corner of the sidebar
    # everything below sits at the bottom of the sidebar (account, data mode, backend status)
    with st.container(key="side_meta"):
        if st.session_state.auth:
            with st.container(key="side_account"):
                a, b = st.columns([4, 1], vertical_alignment="center")
                no = html.escape(st.session_state.auth["student_no"])
                a.markdown(account_html(no), unsafe_allow_html=True)
                if b.button("Log out", icon=":material/logout:", help="Log out", key="logout"):
                    log_out()
        if is_private():
            st.markdown(status_html("lock", "Privacy mode", "No data leaves your environment", "ok"),
                        unsafe_allow_html=True)
        else:
            st.markdown(status_html("shield", "Hybrid mode", "Only 3–5 context chunks are sent to the API",
                                    "info"), unsafe_allow_html=True)
        h = backend_health()
        if h:
            st.markdown(status_html("check_circle", "Backend connected",
                                    f"Default LLM: {html.escape(str(h['provider']))}", "ok"),
                        unsafe_allow_html=True)
            if h.get("models") == "loading":
                st.markdown(status_html("hourglass_top", "Loading models",
                                        "Embedding and reranker models (the first start downloads ~4 GB). "
                                        "Questions will be slow until this finishes.", "warn"),
                            unsafe_allow_html=True)
            elif h.get("models") == "failed":
                st.markdown(status_html("error", "Model loading failed",
                                        html.escape(str(h.get("models_error"))), "error"),
                            unsafe_allow_html=True)
        else:
            st.markdown(status_html("cloud_off", "Backend offline",
                                    f"{html.escape(BASE_URL)} · start it with "
                                    "<code>uvicorn backend.main:app</code>", "error"),
                        unsafe_allow_html=True)
st.navigation(pages).run()