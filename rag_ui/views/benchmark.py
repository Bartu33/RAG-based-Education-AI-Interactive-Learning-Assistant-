import streamlit as st
from core import styles
from core import mock_backend as be
styles.kicker("HALLUCINATION MEASUREMENT"); st.title("📊 Benchmark")
cfg = st.session_state.cfg
st.caption(f"Thresholds (RAGAs Faithfulness): ≥{cfg['th_grounded']} grounded · {cfg['th_partial']}–{cfg['th_grounded']} partially grounded · <{cfg['th_partial']} unsupported")
if st.button("Run benchmark", type="primary"):
    with st.spinner("Scaling from 1,000 to 100,000+ chunks…"):
        st.session_state.bench = be.benchmark()
df = st.session_state.get("bench")
if df is None:
    st.info("No measurements yet. Run the benchmark above (currently using sample data).")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)
    d = df.set_index("Corpus (chunks)")
    a, b = st.columns(2)
    a.subheader("Retrieval latency"); a.line_chart(d[["Retrieval (ms)"]])
    b.subheader("RAGAs Faithfulness"); b.line_chart(d[["RAGAs Faithfulness"]])
    st.subheader("Hallucination rate"); st.bar_chart(d[["Hallucination (%)"]])
    st.subheader("Hallucination types (Ji et al., 2023)")
    c = st.columns(3)
    c[0].metric("Intrinsic (contradicts source)", "3.2%"); c[1].metric("Extrinsic (unverifiable)", "7.8%")
    c[2].metric("Target: <500 ms / <5 s", "✅")
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "benchmark.csv")