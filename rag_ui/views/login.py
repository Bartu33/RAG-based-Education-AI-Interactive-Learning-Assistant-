import streamlit as st
from core import api_client as api

if st.session_state.get("login_notice"):
    st.warning(st.session_state.login_notice)


def signed_in(res: dict) -> None:
    st.session_state.auth = {"token": res["token"], "student_no": res["student_no"]}
    st.session_state.login_notice = None
    st.rerun()


FEATURES = [("forum", "Ask your course", "Answers come from your own slides, notes and lecture videos, with the page or minute."),
            ("quiz", "Practise", "Generated tests in four formats, checked on the spot, in your instructor's style."),
            ("insights", "Track your progress", "Every test and answer is kept in My Results.")]
feats = "".join(f'<div class="login-feat"><div class="i"><span class="material-symbols-rounded">{i}</span></div>'
                f'<div><div class="t">{t}</div><div class="d">{d}</div></div></div>' for i, t, d in FEATURES)
st.markdown('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" />',
            unsafe_allow_html=True)
hero, mid = st.columns([6, 5], gap="large")
hero.markdown(f"""
<div class="login-hero">
  <h1>Interactive <span class="hl">Learning Assistant</span></h1>
  <p class="lead">Log in with your student number to ask questions about your courses, practise with
  generated tests and keep track of your results.</p>
  {feats}
</div>
""", unsafe_allow_html=True)
with mid, st.container(key="login_box"):
    st.markdown("### Welcome")
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
    st.caption(":material/lock: Passwords are stored only as salted hashes on the institution's server.")
