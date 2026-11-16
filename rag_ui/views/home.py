import html
from datetime import date, datetime, timedelta

import streamlit as st
from core import api_client as api
from core.state import cached_courses, cached_documents, log_out, token

ss = st.session_state
TILE_COLORS = ["#1865F2", "#7C5CFA", "#0FA37F", "#E8890C", "#E5484D", "#0E9AA7"]


def history() -> dict | None:
    try:
        return api.history(token())
    except api.ApiError as exc:
        if exc.status == 401:
            log_out("Your session expired, please log in again.")
        return None


def streak(tests: list[dict]) -> int:
    """Days in a row (up to today, or yesterday) on which a test was started."""
    days = {datetime.fromisoformat(t["created"]).astimezone().date() for t in tests}
    day = date.today() if date.today() in days else date.today() - timedelta(days=1)
    n = 0
    while day in days:
        n, day = n + 1, day - timedelta(days=1)
    return n


def go(page: str, **state) -> None:
    """Open a page with its course picker already set (the pickers read these keys)."""
    for k, v in state.items():
        ss[k] = v
    st.switch_page(page)


data = history()
tests = (data or {}).get("tests", [])
summary = (data or {}).get("summary")
try:
    courses = cached_courses()
    docs = cached_documents()
except api.ApiError:
    courses, docs = None, None

# --- greeting + streak + numbers ---
hour = datetime.now().hour
hello = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
st.markdown(f'<div class="dash-hello">{hello}, {html.escape(ss.auth["student_no"])} 👋</div>'
            f'<div class="dash-sub">{date.today():%A, %d %B} · here is where you are.</div>',
            unsafe_allow_html=True)

days = streak(tests)
c0, c1, c2, c3 = st.columns([1.3, 1, 1, 1], gap="medium")
c0.markdown(f'<div class="streak"><div class="flame">{"🔥" if days else "🌱"}</div><div>'
            f'<div class="n">{days} day{"s" if days != 1 else ""}</div>'
            f'<div class="l">{"practice streak, keep it going!" if days else "practise today to start a streak"}'
            f'</div></div></div>', unsafe_allow_html=True)
if summary:
    ch, op = summary["choice"], summary["open"]
    c1.metric("Tests taken", summary["tests"])
    c2.metric("Choice accuracy", f"{100 * ch['correct'] / ch['count']:.0f}%" if ch["count"] else "—")
    c3.metric("Open answers (avg)", f"{op['average']:.0f}/100" if op["average"] is not None else "—")
else:
    c1.metric("Tests taken", "—"); c2.metric("Choice accuracy", "—"); c3.metric("Open answers (avg)", "—")

# --- continue where you left off ---
st.markdown("### Continue where you left off")
left, right = st.columns(2, gap="medium")
with left, st.container(border=True):
    if tests:
        t = tests[0]   # newest first
        s = t["summary"]
        where = t["course"] + (f" · Lecture {t['lecture']}" if t["lecture"] else "")
        st.markdown(f'<div class="continue"><div class="i">📝</div><div><div class="t">Test {t["number"]} · '
                    f'{html.escape(where)}</div><div class="d">{s["answered"]} of {s["questions"]} questions '
                    f'answered</div></div></div>', unsafe_allow_html=True)
        st.page_link("views/history.py", label="Review it in My Results", icon=":material/arrow_forward:")
    else:
        st.markdown('<div class="continue"><div class="i">📝</div><div><div class="t">No tests yet</div>'
                    '<div class="d">Generate your first practice test.</div></div></div>', unsafe_allow_html=True)
        st.page_link("views/question_gen.py", label="Practise now", icon=":material/arrow_forward:")
with right, st.container(border=True):
    last = ss.get("iv_current")   # set by Interactive Learning when a document is open
    if last:
        course, document = last.split("|", 1)
        st.markdown(f'<div class="continue"><div class="i">📖</div><div><div class="t">{html.escape(document)}</div>'
                    f'<div class="d">{html.escape(course)} · last opened document</div></div></div>',
                    unsafe_allow_html=True)
        st.page_link("views/interactive.py", label="Keep reading", icon=":material/arrow_forward:")
    else:
        st.markdown('<div class="continue"><div class="i">📖</div><div><div class="t">Study a document</div>'
                    '<div class="d">Open slides, notes or a lecture video with the study buddy.</div></div></div>',
                    unsafe_allow_html=True)
        st.page_link("views/interactive.py", label="Open Interactive Learning", icon=":material/arrow_forward:")

# --- courses ---
st.markdown("### Your courses")
if courses is None:
    st.warning("The backend is not reachable, so your courses cannot be shown. Start it with "
               "`uvicorn backend.main:app`.")
elif not courses:
    st.info("No course material yet. Start by adding your slides, notes or lecture videos.")
    st.page_link("views/materials.py", label="Upload materials", icon=":material/upload:")
else:
    done = {}
    for t in tests:
        done[t["course"]] = done.get(t["course"], 0) + 1
    for row in range(0, len(courses), 3):
        for i, (col, c) in enumerate(zip(st.columns(3, gap="medium"), courses[row:row + 3])):
            color = TILE_COLORS[(row + i) % len(TILE_COLORS)]
            with col, st.container(key=f"course_{row + i}"):
                lectures = len(c["lectures"])
                st.markdown(f'<div class="course-tile" style="background:{color}"><div class="c">'
                            f'{html.escape(c["course"])}</div><div class="m">{c["documents"]} document{"s" if c["documents"] != 1 else ""} · '
                            f'{lectures} lecture{"s" if lectures != 1 else ""} · {done.get(c["course"], 0)} '
                            f'test{"s" if done.get(c["course"], 0) != 1 else ""}</div></div>', unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                if b1.button("Ask", key=f"ask_{c['course']}", width="stretch"):
                    go("views/qa.py", qa_course=c["course"])
                if b2.button("Practise", key=f"prac_{c['course']}", width="stretch"):
                    go("views/question_gen.py", qg_course=c["course"])
                if b3.button("Study", key=f"study_{c['course']}", width="stretch"):
                    go("views/interactive.py", iv_course=c["course"])

# --- totals ---
st.markdown("### Your study space")
m = st.columns(4, gap="medium")
m[0].metric("Documents", "—" if docs is None else len(docs))
m[1].metric("Indexed chunks", "—" if docs is None else sum(d["chunks"] for d in docs))
m[2].metric("Questions asked", sum(1 for x in ss.chat if x["role"] == "user"))
m[3].metric("Practice questions", len(ss.questions))

st.markdown('<div class="privacy">🔒 <span>Your course files are processed locally. Only a few short, relevant '
            'excerpts are ever sent to the language model.</span></div>', unsafe_allow_html=True)
