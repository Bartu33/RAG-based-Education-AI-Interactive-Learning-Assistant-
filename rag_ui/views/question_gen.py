import json, uuid, streamlit as st
from core import styles
from core.board import fix_markdown
from core import api_client as api
from core.state import QTYPES, course_picker, is_choice, record_attempt, render_evaluation, safe, token
styles.kicker("QUESTION GENERATION"); st.title("📝 Question Generation")
MAX_PER_TYPE, MAX_TOTAL = 10, 20   # same limits as backend/tasks/question_gen.py
FMT_OF = dict(zip(QTYPES, ("mcq", "tf", "short", "concept")))


def match_mix(exams):
    """Fill the per-type boxes with the past exams' mix of question types (same total as now)."""
    mix = {t: sum(e["mix"][FMT_OF[t]] for e in exams) for t in QTYPES}
    total = sum(st.session_state.get(f"qg_n_{t}", 0) for t in QTYPES) or 10
    if not sum(mix.values()):
        return
    share = {t: total * k / sum(mix.values()) for t, k in mix.items()}
    n = {t: int(v) for t, v in share.items()}
    for t in sorted(share, key=lambda t: share[t] - n[t], reverse=True)[:total - sum(n.values())]:
        n[t] += 1  # largest remainders
    for t in QTYPES:
        st.session_state[f"qg_n_{t}"] = min(MAX_PER_TYPE, n[t])


def past_exams(course):
    """Upload / list / remove the course's past exams (instructor's question style)."""
    exams = safe(api.exams, course) or []
    with st.expander(f"🧾 Past exams of {course} ({len(exams)}): questions in the instructor's style"):
        st.caption("Upload old exams of this course. They are analysed once: how the instructor asks, which "
                   "topics come up. Then choose **Like the past exams** below. Exams are only style examples: "
                   "answers still come from the course material, and Q&A never cites them.")
        for e in exams:
            with st.container(border=True):
                h1, h2 = st.columns([5, 1])
                h1.markdown(f"**{e['name']}** · {len(e['questions'])} questions · uploaded {e['created'][:10]}")
                if h2.button("Remove", key=f"del_exam_{e['id']}"):
                    if safe(api.delete_exam, e["id"]) is not None:
                        st.rerun()
                if e["summary"]:
                    st.write(e["summary"])
                if e["style_points"]:
                    st.markdown("\n".join(f"- {p}" for p in e["style_points"]))
                if e["topics"]:
                    st.caption("Topics: " + ", ".join(e["topics"]))
                if e["truncated"]:
                    st.caption("Only the first ~5-6 pages were analysed.")
        with st.form("add_exam", clear_on_submit=True):
            name = st.text_input("Name", placeholder="e.g. Midterm 2023")
            up = st.file_uploader("Exam file", type=["pdf", "docx", "pptx"])
            text = st.text_area("…or paste the questions", height=120,
                                help="Use this for scanned exams, whose PDF has no text.")
            if st.form_submit_button("Analyse and save", type="primary"):
                if not up and not text.strip():
                    st.warning("Choose a file or paste the questions.")
                else:
                    with st.spinner("Reading the exam and learning the instructor's style…"):
                        file = (up.name, up.getvalue(), up.type or "application/octet-stream") if up else None
                        r = safe(api.add_exam, course, name, text, file, st.session_state.cfg["provider"])
                    if r:
                        st.rerun()
    return exams


with st.container(border=True):
    course, lecture = course_picker("qg")
    exams = past_exams(course)
    c1, c2 = st.columns(2)
    diff = c1.select_slider("Difficulty", ["Easy", "Medium", "Hard"], "Medium")
    topic = c2.text_input("Focus topic (optional)", placeholder="e.g. sampling theorem, Fourier transform",
                          help="Questions are generated from the parts of the course about this topic. "
                               "Leave empty for a mix from the whole course / lecture.")
    style = st.radio("Question style", ["Course material", "Like the past exams"], horizontal=True,
                     disabled=not exams, key="qg_style",
                     help="Like the past exams: new questions phrased the way the instructor asks, about the "
                          "topics the exams focus on. Upload exams above first." if exams else
                          "Upload past exams above to use this.")
    chosen = []
    if exams and style == "Like the past exams":
        by_id = {e["id"]: e for e in exams}
        e1, e2 = st.columns([3, 1])
        ids = e1.multiselect("Past exams to imitate", list(by_id), list(by_id),
                             format_func=lambda i: by_id[i]["name"])
        chosen = [by_id[i] for i in ids]
        e2.button("Use their question mix", disabled=not chosen, on_click=match_mix, args=(chosen,),
                  help="Sets the boxes below to the share of each question type in these exams.")
    st.markdown("**How many questions of each type?**")
    defaults = {"Multiple Choice": 3, "True/False": 2, "Short Answer": 0, "Conceptual/Computational": 0}
    for t in QTYPES:
        st.session_state.setdefault(f"qg_n_{t}", defaults[t])
    counts = {t: col.number_input(t, 0, MAX_PER_TYPE, key=f"qg_n_{t}")
              for col, t in zip(st.columns(len(QTYPES)), QTYPES)}
    n = sum(counts.values())
    st.caption(f"Total: **{n}** question{'s' if n != 1 else ''} (at most {MAX_TOTAL} per test)")
    exam_style = style == "Like the past exams"
    if st.button("Generate questions", type="primary",
                 disabled=not 0 < n <= MAX_TOTAL or (exam_style and not chosen)):
        with st.spinner(f"Retrieving context and generating {n} questions…"):
            qs = safe(api.generate_questions, course, lecture, {t: k for t, k in counts.items() if k},
                      diff, st.session_state.cfg["provider"], topic.strip() or None,
                      [e["id"] for e in chosen] if exam_style else None)
        if qs is not None:
            # one generated set = one test in My Results; its questions are stored with it
            test_id = safe(api.create_test, token(), course, lecture, diff, qs) if qs else None
            for q in qs:  # grading uses the same course material; answers are saved under the test
                q.update(course=course, lecture=lecture, test_id=test_id)
            st.session_state.questions = qs
            st.session_state.checked, st.session_state.graded = {}, {}
            for k in [k for k in st.session_state if str(k).startswith(("pick_", "txt_"))]:
                del st.session_state[k]  # fresh answer widgets for the new set
            got = {t: sum(1 for q in qs if q["type"] == t) for t in counts}
            missing = [f"{counts[t] - got[t]} {t}" for t in counts if got[t] < counts[t]]
            if missing:
                st.warning(f"{n - len(qs)} of {n} questions could not be used (the model returned them in an "
                           f"invalid format): {', '.join(missing)}. Generate again to get a full set.")
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
    st.markdown(fix_markdown(q["explanation"])); st.caption(f"📎 {q['source']}")


for i, q in enumerate(qs, 1):
    with st.container(border=True):
        st.markdown(f'<span class="pill">{q["type"]}</span><span class="pill">{q["difficulty"]}</span>', unsafe_allow_html=True)
        st.markdown(f"**{i}.** " + fix_markdown(q["question"]))
        if q.get("style_ref"):
            st.caption(f"🧾 In the style of {q['style_ref']}")
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
                    st.success(fix_markdown(q["correct_answer"])); explanation(q)
                if st.button("Try again", key=f"retry_{q['id']}"):
                    graded.pop(q["id"]); st.session_state.pop(f"txt_{q['id']}", None); st.rerun()
if not qs:
    st.info("No questions generated yet.")
