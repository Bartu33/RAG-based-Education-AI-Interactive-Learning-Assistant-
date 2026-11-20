import streamlit as st
from core import api_client as api
from core.ui_components import APP_NAME, logo_html

# A full-page sign-in (no sidebar), laid out like Khan Academy's: a light introduction on the left,
# the log-in / register card on the right, "How it works" below and the brand bar at the bottom.
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stHeaderLogo"]{ display:none !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{ background:
    radial-gradient(620px 420px at 6% 96%, rgba(52,81,209,.07), transparent 70%),
    radial-gradient(520px 380px at 96% 8%, rgba(124,111,224,.07), transparent 70%),
    linear-gradient(180deg,#F5F7FB 0%,#FFFFFF 70%) !important; }
/* One screen on a computer: brand bar on top, sign-in in the middle (centred in the free height),
   "How it works" at the bottom. The block is exactly the window height (padding included), so the
   page itself never scrolls; shorter windows get a more compact layout (styles.css, section 12). */
@media (min-width: 901px){
  .block-container{ max-width:1400px !important; height:100vh; min-height:0 !important; box-sizing:border-box;
    display:flex; flex-direction:column; padding-top:1.25rem !important; padding-bottom:1.25rem !important; }
  .block-container > [data-testid="stVerticalBlock"]{ flex:1 1 auto; min-height:0; gap:0 !important; }
  .block-container > [data-testid="stVerticalBlock"] > :has(.lg-intro){ margin-top:auto !important; margin-bottom:auto !important;
    padding:.9rem 0; }
}
</style>
""", unsafe_allow_html=True)

# --- brand bar on top (logo + name), then the sign-in, then "How it works" ---
with st.container(key="lg_top"):
    st.markdown(f'<div class="lg-top"><div class="lg-bar">{logo_html(32)}<div class="name">{APP_NAME}</div>'
                '<div class="tag">RAG-based Education AI</div></div>'
                '<span>Hybrid RAG · answers grounded in approved course material</span></div>', unsafe_allow_html=True)

if st.session_state.get("login_notice"):
    st.warning(st.session_state.login_notice)


def signed_in(res: dict) -> None:
    st.session_state.auth = {"token": res["token"], "student_no": res["student_no"]}
    st.session_state.login_notice = None
    st.rerun()


FEATURES = [("forum", "Ask your course", "Answers from your own slides, notes and lecture videos, with the page or minute they come from."),
            ("quiz", "Practise", "Generated tests in four formats, checked on the spot, even in your instructor's style."),
            ("insights", "Track your progress", "Every test and answer is kept, so you see what to revise.")]
feats = "".join(f'<div class="lg-feat"><div class="i"><span class="mi">{i}</span></div>'
                f'<div><div class="t">{t}</div><div class="d">{d}</div></div></div>' for i, t, d in FEATURES)

intro, card = st.columns([7, 5], gap="large", vertical_alignment="center")
intro.markdown(f"""
<div class="lg-intro">
  <div class="lg-eyebrow">For students of your institution</div>
  <h1>Learn from your own course, <span class="hl">at your own pace.</span></h1>
  <p class="lead">An assistant that only reads the materials your instructors approved: ask about them,
  study a document or a lecture video with a study buddy, and practise with tests made from them.</p>
  {feats}
  <div class="lg-note"><span class="mi">lock</span> Your files stay on the institution's server. Only a few short
  excerpts are sent to the language model.</div>
</div>
""", unsafe_allow_html=True)

with card, st.container(key="login_box"):
    st.markdown(f'<div style="margin-bottom:1rem">{logo_html(52, large=True)}</div>'
                '<div class="lg-title">Welcome back</div>'
                '<div class="lg-sub">Log in with your student number, or create an account.</div>',
                unsafe_allow_html=True)
    log_in, sign_up = st.tabs(["Log in", "Register"])
    with log_in:
        with st.form("login", border=False):
            no = st.text_input("Student number", placeholder="e.g. 220201032")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                try:
                    signed_in(api.login(no, pw))
                except api.ApiError as exc:
                    st.error(str(exc))
        st.caption("New here? Choose **Register** above.")
    with sign_up:
        with st.form("register", border=False):
            no = st.text_input("Student number", help="3-32 letters or digits, e.g. 220201032",
                               placeholder="e.g. 220201032")
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

# --- below the sign-in: how it works ---
STEPS = [("upload_file", "Your instructor adds the material", "Slides, PDFs, notes and lecture recordings are indexed on the institution's server."),
         ("search", "You ask or study", "Every answer is searched in that material and shows the page, slide or minute it is based on."),
         ("task_alt", "You practise and check", "Generated tests are checked on the spot; your results show what to revise.")]
steps = "".join(f'<div class="lg-step"><div class="n">{n}</div><div class="i"><span class="mi">{i}</span></div>'
                f'<div class="t">{t}</div><div class="d">{d}</div></div>' for n, (i, t, d) in enumerate(STEPS, 1))
with st.container(key="lg_how"):
    st.markdown(f'<div class="lg-how"><div class="lg-how-title">How it works</div><div class="lg-steps">{steps}</div></div>',
                unsafe_allow_html=True)
