import pandas as pd, streamlit as st
from core import styles
from core import api_client as api
from core.state import backend_config, safe
styles.kicker("HALLUCINATION MEASUREMENT"); st.title("📊 Benchmark")
cfg = st.session_state.cfg
st.caption(f"Thresholds (RAGAs Faithfulness): ≥{cfg['th_grounded']} grounded · {cfg['th_partial']}–{cfg['th_grounded']} partially grounded · <{cfg['th_partial']} unsupported")
all_sizes = (safe(backend_config) or {}).get("benchmark_sizes", [1000, 10000, 50000, 100000])
sizes = st.multiselect("Corpus sizes (chunks)", all_sizes, default=all_sizes[:2])
st.caption("Uses the questions in data/benchmark_questions.json. Each size makes ~2–3 LLM calls per question, so large runs take a while.")
if st.button("Run benchmark", type="primary", disabled=not sizes):
    with st.spinner(f"Building corpora of {', '.join(f'{s:,}' for s in sorted(sizes))} chunks and answering the questions…"):
        rows = safe(api.benchmark, sorted(sizes), cfg["provider"])
    if rows is not None:
        st.session_state.bench = pd.DataFrame(rows)
df = st.session_state.get("bench")
if df is None or df.empty:
    st.info("No measurements yet. Run the benchmark above.")
else:
    st.dataframe(df, width="stretch", hide_index=True)
    d = df.set_index("Corpus (chunks)")
    a, b = st.columns(2)
    a.subheader("Retrieval latency"); a.line_chart(d[["Retrieval (ms)"]])
    b.subheader("RAGAs Faithfulness"); b.line_chart(d[["RAGAs Faithfulness"]])
    st.subheader("Hallucination rate"); st.bar_chart(d[["Hallucination (%)"]])
    st.subheader("Hallucination types (Ji et al., 2023)")
    c = st.columns(3)
    c[0].metric("Intrinsic (contradicts source)", f"{df['Intrinsic (%)'].mean():.1f}%")
    c[1].metric("Extrinsic (unverifiable)", f"{df['Extrinsic (%)'].mean():.1f}%")
    met = df["Retrieval (ms)"].max() < 500 and df["Answer (s)"].max() < 5
    c[2].metric("Target: <500 ms / <5 s", "✅" if met else "❌")
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "benchmark.csv")
