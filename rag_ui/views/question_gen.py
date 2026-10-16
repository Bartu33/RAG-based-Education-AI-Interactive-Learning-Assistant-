import json, uuid, streamlit as st
from core import styles
from core import api_client as api
from core.state import QTYPES, course_picker, is_choice, record_attempt, render_evaluation, safe, token
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
            # one generated set = one test in My Results; its questions are stored with it
            test_id = safe(api.create_test, token(), course, lecture, diff, qs) if qs else None
            for q in qs:  # grading uses the same course material; answers are saved under the test
                q.update(course=course, lecture=lecture, test_id=test_id)
            st.session_state.questions = qs
            st.session_state.checked, st.session_state.graded = {}, {}
            for k in [k for k in st.session_state if str(k).startswith(("pick_", "txt_"))]:
                del st.session_state[k]  # fresh answer widgets for the new set
            if len(qs) < n:
                st.warning(f"{n - len(qs)} of {n} questions were dropped because the model returned them in an invalid format.")
qs = st.session_state.questions
checked, graded = st.session_state.checked, st.session_state.graded

if qs:
    top = st.columns([3, 1])
    choice_qs = [q for q in qs if is_choice(q)]
    open_scores = [graded[q["id"]]["total"] for q in qs if q["id"] in graded]
    parts = []
    if choice_qs:
        right = sum(1 for q in choice_qs if checked.get(q["id"], {}).get("correct"))
        parts.append(f"Choice questions: {right} of {len(choice_qs)} correct "
                     f"({sum(1 for q in choice_qs if q['id'] in checked)} checked)")
    if len(choice_qs) < len(qs):
        avg = f", average {sum(open_scores) / len(open_scores):.0f}/100" if open_scores else ""
        parts.append(f"Open questions: {len(open_scores)} of {len(qs) - len(choice_qs)} graded{avg}")
    top[0].caption(" · ".join(parts))
    top[1].download_button("⬇️ Download as JSON", json.dumps(qs, ensure_ascii=False, indent=2), "questions.json")


def explanation(q):
    st.write(q["explanation"]); st.caption(f"📎 {q['source']}")


for i, q in enumerate(qs, 1):
    with st.container(border=True):
        st.markdown(f'<span class="pill">{q["type"]}</span><span class="pill">{q["difficulty"]}</span>', unsafe_allow_html=True)
        st.markdown(f"**{i}. {q['question']}**")
        if is_choice(q):
            # graded locally: the correct answer is part of the question, so no LLM call is needed
            options = q["options"] or ["True", "False"]
            labels = [f"{k}) {o}" for k, o in zip("ABCD", options)] if q["options"] else options
            done = checked.get(q["id"])
            pick = st.radio("Your answer", labels, index=None, key=f"pick_{q['id']}",
                            disabled=done is not None, label_visibility="collapsed")
            if done is None:
                if st.button("Check", key=f"check_{q['id']}", type="primary", disabled=pick is None):
                    chosen = options[labels.index(pick)]
                    correct = chosen.strip().lower() == str(q["correct_answer"]).strip().lower()
                    checked[q["id"]] = {"correct": correct}
                    record_attempt(q, pick, is_correct=correct)
                    st.rerun()
            else:
                if done["correct"]:
                    st.success("✅ Correct")
                else:
                    key = str(q["correct_answer"]).strip().lower()
                    right = next((l for o, l in zip(options, labels) if o.strip().lower() == key), q["correct_answer"])
                    st.error(f"❌ Not quite. The correct answer is **{right}**.")
                explanation(q)
                if st.button("Try again", key=f"retry_{q['id']}"):
                    checked.pop(q["id"]); st.session_state.pop(f"pick_{q['id']}", None); st.rerun()
        else:
            # open-ended: graded with the rubric, using the generated model answer as a reference
            result = graded.get(q["id"])
            text = st.text_area("Your answer", key=f"txt_{q['id']}", height=110, disabled=result is not None,
                                placeholder="Write your answer here…", label_visibility="collapsed")
            if result is None:
                if st.button("Evaluate", key=f"eval_{q['id']}", type="primary"):
                    if not text.strip():
                        st.warning("Write your answer first.")
                    else:
                        with st.spinner("Scoring with the rubric…"):
                            r = safe(api.evaluate, q["question"], text, q["course"], q["lecture"],
                                     st.session_state.cfg["provider"], q["correct_answer"])
                        if r:
                            r["rid"] = uuid.uuid4().hex[:12]
                            graded[q["id"]] = r
                            record_attempt(q, text, result=r)
                            st.rerun()
            else:
                render_evaluation(result, result.get("rid"), q["question"])
                with st.expander("Model answer"):
                    st.success(q["correct_answer"]); explanation(q)
                if st.button("Try again", key=f"retry_{q['id']}"):
                    graded.pop(q["id"]); st.session_state.pop(f"txt_{q['id']}", None); st.rerun()
if not qs:
    st.info("No questions generated yet.")
