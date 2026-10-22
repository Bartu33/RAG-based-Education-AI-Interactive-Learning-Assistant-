import streamlit as st
from core import benchmark_panel, styles
from core.state import PRESETS, PROVIDERS, backend_config, backend_defaults, preset_of, safe
c = st.session_state.cfg
head, switch = st.columns([3, 1], vertical_alignment="bottom")
with head:
    styles.kicker("CONFIGURATION"); st.title("⚙️ Settings")
view = switch.segmented_control("View", ["General", "Advanced"], default="General", required=True,
                                key="settings_view", label_visibility="collapsed")

# the provider stays on both views: switch when the daily tokens of one run out
st.subheader("Language model")
c["provider"] = st.radio("Provider", PROVIDERS, index=PROVIDERS.index(c["provider"]),
                         help="Groq and OpenAI run in the cloud (an API key in .env). Ollama runs on this "
                              "computer: nothing leaves it, but it is slower. If one provider's daily limit "
                              "is used up, pick another one here.")

if view == "General":
    st.subheader("Answer quality")
    INFO = {
        "Low": ("⚡ Fast", "Fewest course excerpts and no grounding check. Quickest answers and the "
                          "fewest tokens: good when the daily limit is running low."),
        "Medium": ("⚖️ Balanced", "The recommended setting: enough excerpts for most questions, and every "
                                 "Q&A answer is checked against the material (🟢 / 🟡 / 🔴 label)."),
        "High": ("🎯 Thorough", "The most excerpts: fuller answers to broad questions that span several "
                               "pages, but slower and uses the most tokens."),
    }
    now = preset_of(c)
    pick = st.segmented_control("Answer quality", list(PRESETS), default=now,
                                format_func=lambda p: f"{INFO[p][0]} · {p}", key=f"preset_{now}",
                                label_visibility="collapsed")
    if pick and pick != now:
        c.update(PRESETS[pick]); st.rerun()
    if now:
        p = PRESETS[now]
        chunk = (safe(backend_config) or {}).get("chunk_size", 500)
        tokens = p["top_k_context"] * chunk * (2 if p["score_faithfulness"] else 1)
        with st.container(border=True):
            st.markdown(f"**{INFO[now][0]}**: {INFO[now][1]}")
            st.caption(f"Q&A: {p['top_k_context']} excerpts out of the {p['top_k_candidates']} best matches · "
                       f"grounding check {'on' if p['score_faithfulness'] else 'off'} · "
                       f"Study Buddy: {p['study_top_k']} excerpts · about {tokens:,} tokens of course text "
                       "per Q&A question")
    else:
        st.info("Custom settings are in use (changed in **Advanced**). Pick a level above to go back to a preset.")
    st.caption("Everything else (thresholds, benchmark) is under **Advanced**, top right.")

else:
    retrieval, bench = st.tabs(["🔧 Retrieval and thresholds", "📊 Benchmark"])
    with retrieval:
        st.subheader("Retrieval")
        a, b = st.columns(2)
        c["top_k_candidates"] = a.slider("Candidate chunks (similarity search)", 5, 30, c["top_k_candidates"])
        c["top_k_context"] = b.slider("Context chunks (after reranking)", 1, 8, c["top_k_context"])
        a, b = st.columns(2)
        c["study_top_k"] = a.slider("Study Buddy excerpts per question", 1, 8, c["study_top_k"])
        c["score_faithfulness"] = b.toggle("Grounding check on Q&A answers", c["score_faithfulness"],
                                           help="One extra LLM call per answer: the 🟢 / 🟡 / 🔴 label.")
        st.subheader("Thresholds")
        c["conf_threshold"] = st.slider("STT confidence threshold (flags transcript segments to correct)", 0.0, 1.0, c["conf_threshold"], 0.05)
        t1, t2 = st.columns(2)
        c["th_grounded"] = t1.number_input("Grounded threshold", 0.0, 1.0, c["th_grounded"], 0.05)
        c["th_partial"] = t2.number_input("Partially grounded threshold", 0.0, 1.0, c["th_partial"], 0.05)
        st.subheader("Indexing (config.yaml)")
        bk = safe(backend_config)
        if bk:
            st.caption("These are fixed when materials are indexed. Change them in config.yaml and re-index; "
                       "the index refuses to mix embedding models.")
            st.table({"Embedding model": [bk["embedding_model"]], "Reranker": [bk["reranker_model"]],
                      "Chunk size (tokens)": [bk["chunk_size"]], "Overlap (tokens)": [bk["chunk_overlap"]],
                      "Video STT": [f"faster-whisper ({bk['whisper_model']})"]})
        if st.button("Reset to defaults"):
            st.session_state.cfg = backend_defaults(); st.rerun()
    with bench:
        benchmark_panel.render()
