import streamlit as st
from core import styles

styles.kicker("RAG-based Education AI")

st.markdown("""
<div class="hero">
  <h1>Study smarter with <span>your own course materials</span></h1>
  <p>Ask questions and get answers with exact page, slide and video-timestamp references.
     Practice with auto-generated questions and get feedback on your answers.</p>
</div>
""", unsafe_allow_html=True)

# --- yardımcı: kayıt alan adları çeviriden etkilenmesin ---
def _chunks(d):
    for k in ("Chunks", "chunks", "Parça"):
        if k in d:
            return d[k]
    return 0

docs = st.session_state.docs

# --- ana özellikler ---
cards = [
    ("card_qa", "💬", "Ask a question", "Get answers grounded in your course, with sources you can check.", "views/qa.py"),
    ("card_gen", "📝", "Practice questions", "Multiple choice, true/false, short answer and conceptual questions.", "views/question_gen.py"),
    ("card_eval", "✅", "Check my answer", "Rubric-based score with clear feedback on what you missed.", "views/evaluation.py"),
    ("card_mat", "📚", "Course materials", "Upload slides, notes, textbooks and lecture videos.", "views/materials.py"),
]
for col, (key, icon, title, desc, page) in zip(st.columns(4, gap="medium"), cards):
    with col:
        with st.container(key=key):
            st.markdown(f'<div class="ico">{icon}</div><div class="card-title">{title}</div>'
                        f'<div class="card-desc">{desc}</div>', unsafe_allow_html=True)
            st.page_link(page, label="Open →")

# --- nasıl çalışır ---
st.markdown("### How it works")
steps = [("1", "Add your materials", "PDFs, slides, notes and lecture videos are processed on your institution's own hardware."),
         ("2", "Ask or practice", "Questions are answered only from approved course content, never from the open web."),
         ("3", "Verify and improve", "Every answer shows its sources and a reliability score, so you know what to trust.")]
for col, (n, t, d) in zip(st.columns(3, gap="medium"), steps):
    col.markdown(f'<div class="step"><b class="n">{n}</b><div class="t">{t}</div><div class="d">{d}</div></div>',
                 unsafe_allow_html=True)

# --- istatistikler ---
st.markdown("### Your study space")
m = st.columns(4, gap="medium")
m[0].metric("Documents", len(docs))
m[1].metric("Indexed chunks", sum(_chunks(d) for d in docs))
m[2].metric("Questions asked", sum(1 for x in st.session_state.chat if x["role"] == "user"))
m[3].metric("Practice questions", len(st.session_state.questions))

if not docs:
    st.info("Nothing here yet. Start by adding your course materials.")
    st.page_link("views/materials.py", label="Upload materials →")

st.markdown('<div class="privacy">🔒 <span>Your course files are processed locally. Only a few short, relevant '
            'excerpts are ever sent to the language model.</span></div>', unsafe_allow_html=True)