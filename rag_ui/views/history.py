from datetime import datetime
import pandas as pd, streamlit as st
from core import styles
from core.board import fix_markdown
from core import api_client as api
from core.state import log_out, token
styles.kicker("PRACTICE HISTORY"); st.title("📒 My Results")

try:
    data = api.history(token())
except api.ApiError as exc:
    if exc.status == 401:
        log_out("Your session expired, please log in again.")
    st.error(f"Backend error: {exc}"); st.stop()
tests, total = data["tests"], data["summary"]
if not tests:
    st.info("No tests yet. Generate questions on the Question Generation page; every set you generate "
            "appears here as a test with your answers and the correct ones.")
    st.page_link("views/question_gen.py", label="Practice questions →")
    st.stop()


def local(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone().strftime("%d %b %Y, %H:%M")


def results_line(s: dict) -> str:
    parts = [f"{s['answered']}/{s['questions']} answered"]
    if s["choice"]["count"]:
        parts.append(f"{s['choice']['correct']}/{s['choice']['count']} correct")
    if s["open"]["average"] is not None:
        parts.append(f"open avg {s['open']['average']:.0f}/100")
    return " · ".join(parts)


# --- overall ---
m = st.columns(3)
m[0].metric("Tests", total["tests"], f"{total['answered']} of {total['questions']} questions answered", delta_color="off")
c = total["choice"]
m[1].metric("Choice accuracy", f"{100 * c['correct'] / c['count']:.0f}%" if c["count"] else "—",
            f"{c['correct']} of {c['count']} correct" if c["count"] else None, delta_color="off")
o = total["open"]
m[2].metric("Open questions (avg)", f"{o['average']:.0f}/100" if o["average"] is not None else "—",
            f"{o['count']} graded" if o["count"] else None, delta_color="off")
courses = sorted({t["course"] for t in tests})
course = st.selectbox("Course", ["All courses"] + courses) if len(courses) > 1 else "All courses"


def show_question(i: int, q: dict) -> None:
    a = q["attempt"]
    if a is None:
        mark = "⚪"
    elif a["is_correct"] is not None:
        mark = "✅" if a["is_correct"] else "❌"
    else:
        mark = "🟢" if a["score"] >= 70 else "🟡" if a["score"] >= 50 else "🔴"
    score = f" · {a['score']}/100" if a and a["score"] is not None else ""
    st.markdown(f"{mark} **{i}.** " + fix_markdown(q["question"]))
    st.caption(f"{q['type']}{score}" + (f" · answered {a['tries']} times, latest shown" if a and a["tries"] > 1 else ""))
    if q.get("options"):
        st.write("  \n".join(f"{k}) {o}" for k, o in zip("ABCD", q["options"])))
    left, right = st.columns(2)
    with left:
        st.markdown("**Your answer**")
        if a is None:
            st.info("Not answered")
        else:
            ok = a["is_correct"] if a["is_correct"] is not None else a["score"] >= 70
            weak = a["is_correct"] is None and 50 <= a["score"] < 70
            (st.success if ok else st.warning if weak else st.error)(a["student_answer"])
    with right:
        st.markdown("**Correct answer**")
        st.success(fix_markdown(q["correct_answer"]))
    if q.get("explanation"):
        st.markdown(fix_markdown(q["explanation"]))
    if a and a["feedback"]:
        st.markdown("**Feedback you received**")
        st.write("  \n".join(f"• {f}" for f in a["feedback"]))
    if q.get("source"):
        st.caption(f"📎 {q['source']}")


# --- tests, newest first ---
for t in tests:
    if course != "All courses" and t["course"] != course:
        continue
    where = t["course"] + (f" · Lecture {t['lecture']}" if t["lecture"] else "")
    label = f"📝 Test {t['number']} · {where} · {local(t['created'])} — {results_line(t['summary'])}"
    with st.expander(label):
        st.caption(f"{len(t['questions'])} questions · {t['difficulty'] or ''}")
        for i, q in enumerate(t["questions"], 1):
            if i > 1:
                st.divider()
            show_question(i, q)

# --- export / clear ---
st.divider()
rows = [{"test": t["number"], "created": local(t["created"]), "course": t["course"], "lecture": t["lecture"],
         "question": q["question"], "type": q["type"],
         "your answer": q["attempt"]["student_answer"] if q["attempt"] else "",
         "correct answer": q["correct_answer"],
         "correct": q["attempt"]["is_correct"] if q["attempt"] else None,
         "score": q["attempt"]["score"] if q["attempt"] else None}
        for t in tests for q in t["questions"]]
c1, c2 = st.columns(2)
c1.download_button("⬇️ Download as CSV", pd.DataFrame(rows).to_csv(index=False), "my_results.csv")
with c2.popover("🗑️ Clear my history"):
    st.write(f"This deletes all {len(tests)} tests and your answers.")
    if st.button("Delete", type="primary"):
        try:
            api.clear_history(token()); st.rerun()
        except api.ApiError as exc:
            st.error(str(exc))
