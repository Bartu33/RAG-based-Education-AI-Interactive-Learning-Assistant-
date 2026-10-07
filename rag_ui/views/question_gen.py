import json, streamlit as st
from core import backend as be, styles
styles.kicker("QUESTION GENERATION"); st.title("📝 Question Generation")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    course = c1.selectbox("Course", list(be.COURSES), key="qg_course")
    lecture = c2.selectbox("Lecture number", range(1, 15), index=4, key="qg_lec")
    diff = c3.select_slider("Difficulty", ["Easy", "Medium", "Hard"], "Medium")
    types = st.multiselect("Question formats", be.QTYPES, default=be.QTYPES[:2])
    n = st.slider("Number of questions", 1, 20, 5)
    if st.button("Generate questions", type="primary", disabled=not types):
        with st.spinner("Retrieving context and generating questions…"):
            st.session_state.questions = be.generate_questions(course, lecture, types, n, diff)
qs = st.session_state.questions
if qs:
    st.download_button("⬇️ Download as JSON", json.dumps(qs, ensure_ascii=False, indent=2), "questions.json")
for i, q in enumerate(qs, 1):
    with st.container(border=True):
        st.markdown(f'<span class="pill">{q["type"]}</span><span class="pill">{q["difficulty"]}</span>', unsafe_allow_html=True)
        st.markdown(f"**{i}. {q['question']}**")
        if q["options"]:
            for k, o in zip("ABCD", q["options"]): st.write(f"{k}) {o}")
        with st.expander("Answer and explanation"):
            st.success(q["correct_answer"]); st.write(q["explanation"]); st.caption(f"📎 {q['source']}")
        if st.button("Answer this question →", key=q["id"]):
            st.session_state.eval_target = q; st.switch_page("views/evaluation.py")
if not qs:
    st.info("No questions generated yet.")