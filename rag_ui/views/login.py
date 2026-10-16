import streamlit as st
from core import styles
from core import api_client as api

styles.kicker("RAG-based Education AI")
st.markdown("""
<div class="hero">
  <h1>Interactive <span>Learning Assistant</span></h1>
  <p>Log in with your student number to ask questions about your courses, practise with generated
     tests and keep track of your results.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("login_notice"):
    st.warning(st.session_state.login_notice)


def signed_in(res: dict) -> None:
    st.session_state.auth = {"token": res["token"], "student_no": res["student_no"]}
    st.session_state.login_notice = None
    st.rerun()


_, mid, _ = st.columns([1, 2, 1])
with mid:
    log_in, sign_up = st.tabs(["Log in", "Register"])
    with log_in:
        with st.form("login"):
            no = st.text_input("Student number")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                try:
                    signed_in(api.login(no, pw))
                except api.ApiError as exc:
                    st.error(str(exc))
    with sign_up:
        with st.form("register"):
            no = st.text_input("Student number", help="3-32 letters or digits, e.g. 220201032")
            pw = st.text_input("Password", type="password", help="At least 6 characters")
            pw2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create account", type="primary", width="stretch"):
                if pw != pw2:
                    st.error("The two passwords do not match.")
                else:
                    try:
                        signed_in(api.register(no, pw))
                    except api.ApiError as exc:
                        st.error(str(exc))
    st.caption("🔒 Passwords are stored only as salted hashes on the institution's server.")
