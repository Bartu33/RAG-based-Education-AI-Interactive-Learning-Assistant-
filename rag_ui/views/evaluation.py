import uuid
import streamlit as st
from core import styles
from core import api_client as api
from core.state import course_picker, render_evaluation, safe
styles.kicker("ANSWER EVALUATION"); st.title("✅ Answer Evaluation")
st.caption("Generated questions are answered and graded right under each question on the Question Generation page. "
           "Here you can write your own question.")
course, lecture = course_picker("ev")
question = st.text_input("Question")
answer = st.text_area("Your answer (free text)", height=160)
# not disabled while empty: a text box only reports its value when it loses focus, so a disabled
# button would swallow the first click right after typing
if st.button("Evaluate", type="primary"):
    if not (question.strip() and answer.strip()):
        st.warning("Write a question and your answer first.")
        st.stop()
    with st.spinner("Scoring with the rubric…"):
        r = safe(api.evaluate, question, answer, course, lecture, st.session_state.cfg["provider"])
    if r:
        st.session_state.eval_history.append({"Question": question, "Score": r["total"]})
        st.session_state.own_eval = {**r, "rid": uuid.uuid4().hex[:12], "question": question}
own = st.session_state.get("own_eval")
if own:  # kept across reruns so the 👍/👎 below it stays clickable
    render_evaluation(own, own["rid"], own["question"])
if st.session_state.eval_history:
    with st.expander("Previous attempts"):
        st.dataframe(st.session_state.eval_history, width="stretch", hide_index=True)
