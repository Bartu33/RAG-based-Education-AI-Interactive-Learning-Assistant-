import streamlit as st
from core import styles
from core import api_client as api
from core.state import course_picker, safe
styles.kicker("ANSWER EVALUATION"); st.title("✅ Answer Evaluation")
tgt = st.session_state.eval_target
qs = st.session_state.questions
src = st.radio("Question source", ["Generated questions", "Write my own"], horizontal=True)
if src == "Generated questions" and qs:
    ids = [q["id"] for q in qs]
    idx = ids.index(tgt["id"]) if tgt and tgt["id"] in ids else 0
    pick = st.selectbox("Question", qs, index=idx, format_func=lambda q: q["question"])
    question, course, lecture = pick["question"], pick["course"], pick["lecture"]
    st.caption(f"Graded against {course}" + (f" · lecture {lecture}" if lecture else ""))
else:
    if src == "Generated questions":
        st.info("No generated questions yet, so write your own below.")
    course, lecture = course_picker("ev")
    question = st.text_input("Question", value=tgt["question"] if tgt else "")
answer = st.text_area("Your answer (free text)", height=160)
if st.button("Evaluate", type="primary", disabled=not (question and answer.strip())):
    with st.spinner("Scoring with the rubric…"):
        r = safe(api.evaluate, question, answer, course, lecture, st.session_state.cfg["provider"])
    if r:
        st.session_state.eval_history.append({"Question": question, "Score": r["total"]})
        c1, c2 = st.columns([1, 2])
        c1.metric("Overall score", f"{r['total']} / 100")
        with c2:
            for d in r["dims"]:
                st.write(f"**{d['name']}** · weight {int(d['weight']*100)}% · {int(d['score']*100)}/100")
                st.progress(d["score"])
        st.subheader("Detailed feedback")
        for f in r["feedback"]: st.write(f)
        st.caption(f"📎 Reference: {r['source']}")
        if r["follow_up_question"]:
            st.info(f"🎯 **Follow-up practice:** {r['follow_up_question']}")
if st.session_state.eval_history:
    with st.expander("Previous attempts"):
        st.dataframe(st.session_state.eval_history, width="stretch", hide_index=True)
