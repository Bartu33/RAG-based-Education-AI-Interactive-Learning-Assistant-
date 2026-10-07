import streamlit as st
from core.state import init_state, is_private
from core import styles

st.set_page_config(page_title="Interactive Learning Assistant", page_icon="🎓", layout="wide")
init_state(); styles.inject()

pages = [
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/materials.py", title="Materials", icon="📚"),
    st.Page("views/qa.py", title="Q&A", icon="💬"),
    st.Page("views/question_gen.py", title="Question Generation", icon="📝"),
    st.Page("views/evaluation.py", title="Answer Evaluation", icon="✅"),
    st.Page("views/benchmark.py", title="Benchmark", icon="📊"),
    st.Page("views/settings.py", title="Settings", icon="⚙️"),
]
with st.sidebar:
    st.markdown("### 🎓 Learning Assistant")
    st.caption("Hybrid RAG · Privacy-focused")
    if is_private():
        st.success("🔒 Privacy mode: no data leaves your environment")
    else:
        st.info("🛡️ Hybrid mode: only 3–5 context chunks are sent to the API")
st.navigation(pages).run()