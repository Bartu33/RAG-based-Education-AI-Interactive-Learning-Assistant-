import streamlit as st
from core import styles
from core import api_client as api
from core.state import safe
styles.kicker("USABILITY & SATISFACTION"); st.title("📋 Feedback")

# System Usability Scale (Brooke, 1996): the standard 10 items, "system" = this assistant
SUS = [
    "I think that I would like to use this assistant frequently.",
    "I found the assistant unnecessarily complex.",
    "I thought the assistant was easy to use.",
    "I think that I would need the support of a technical person to be able to use this assistant.",
    "I found the various functions in this assistant were well integrated.",
    "I thought there was too much inconsistency in this assistant.",
    "I would imagine that most people would learn to use this assistant very quickly.",
    "I found the assistant very cumbersome to use.",
    "I felt very confident using the assistant.",
    "I needed to learn a lot of things before I could get going with this assistant.",
]
SCALE = [1, 2, 3, 4, 5]

with st.container(border=True):
    st.subheader("Usability survey (SUS)")
    st.caption("10 short statements · 1 = strongly disagree, 5 = strongly agree · takes about a minute")
    if st.session_state.get("sus_done"):
        d = st.session_state.sus_done
        st.success(f"Thank you! Your usability score: **{d['score']:.1f} / 100** ({d['grade']}).")
        if st.button("Fill in again"):
            st.session_state.sus_done = None; st.rerun()
    else:
        with st.form("sus"):
            answers = [st.radio(f"{i}. {text}", SCALE, index=None, horizontal=True, key=f"sus_{i}")
                       for i, text in enumerate(SUS, 1)]
            comment = st.text_area("Anything else you would like to tell us? (optional)", height=80)
            if st.form_submit_button("Submit", type="primary"):
                if None in answers:
                    st.warning(f"Please answer all statements ({answers.count(None)} left).")
                else:
                    res = safe(api.submit_sus, answers, comment)
                    if res:
                        st.session_state.sus_done = res; st.rerun()

st.subheader("Results so far")
s = safe(api.feedback_summary)
if s:
    r, sus = s["ratings"], s["sus"]
    c = st.columns(3)
    def pct(k):
        return "—" if r[k]["positive_pct"] is None else f"{r[k]['positive_pct']:.0f}% 👍"
    c[0].metric("Q&A answers", pct("qa"), f"{r['qa']['count']} ratings", delta_color="off")
    c[1].metric("Answer feedback", pct("evaluation"), f"{r['evaluation']['count']} ratings", delta_color="off")
    c[2].metric("SUS (mean)", "—" if sus["mean"] is None else f"{sus['mean']:.1f}",
                f"{sus['count']} surveys" + (f" · {sus['grade']}" if sus["grade"] else ""), delta_color="off")
    st.caption("SUS 68 is the commonly cited average; above 80 is excellent (Bangor et al., 2009). "
               "Ratings are stored in data/feedback/ on the backend.")
    if s["comments"]:
        with st.expander(f"Latest comments ({len(s['comments'])})"):
            for cm in reversed(s["comments"]):
                st.write(f"{'👍' if cm['up'] else '👎'} `{cm['kind']}` {cm['comment']}")
