import json, streamlit as st
from core import styles
from core import api_client as api
from core.state import QTYPES, course_picker, safe
styles.kicker("QUESTION GENERATION"); st.title("📝 Question Generation")
with st.container(border=True):
    course, lecture = course_picker("qg")
    c1, c2 = st.columns(2)
    diff = c1.select_slider("Difficulty", ["Easy", "Medium", "Hard"], "Medium")
    n = c2.slider("Number of questions", 1, 20, 5)
    types = st.multiselect("Question formats", QTYPES, default=QTYPES[:2])
    if st.button("Generate questions", type="primary", disabled=not types):
        with st.spinner("Retrieving context and generating questions…"):
            qs = safe(api.generate_questions, course, lecture, types, n, diff, st.session_state.cfg["provider"])
        if qs is not None:
            for q in qs:  # remembered so the evaluation page grades against the same material
                q.update(course=course, lecture=lecture)
            st.session_state.questions = qs
            if len(qs) < n:
                st.warning(f"{n - len(qs)} of {n} questions were dropped because the model returned them in an invalid format.")
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
