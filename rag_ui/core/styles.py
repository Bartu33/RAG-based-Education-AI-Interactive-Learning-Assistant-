from pathlib import Path

import streamlit as st

# The whole design system lives in rag_ui/assets/styles.css (tokens in :root). This module only
# loads it; inject(), kicker() and beyond_box() keep their names and signatures.
CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "styles.css"


@st.cache_data(show_spinner=False)
def _read_css(path: str, mtime: float) -> str:
    """styles.css as one <style> block. `mtime` is only part of the cache key, so an edited file is
    picked up on the next rerun without restarting Streamlit."""
    return "<style>\n" + Path(path).read_text(encoding="utf-8") + "\n</style>"


def inject():
    if CSS_PATH.is_file():
        st.markdown(_read_css(str(CSS_PATH), CSS_PATH.stat().st_mtime), unsafe_allow_html=True)


def kicker(t):
    st.markdown(f'<div class="kicker">{t}</div>', unsafe_allow_html=True)


def beyond_box(text: str, key: str, source: str = "course materials") -> None:
    """The part of an answer that does NOT come from the course material, in a lavender box."""
    import streamlit as st
    from core.board import fix_markdown   # LaTeX \( \) / \[ \] -> $ / $$ so the formulas render
    with st.container(key=f"beyond_{key}"):
        st.markdown('<div class="beyond-head"><span class="mi">lightbulb</span> Beyond the course material</div>'
                    f'<div class="beyond-note">General knowledge and reasoning, not taken from your '
                    f'{source}. Double-check it with the lecture notes or your instructor.</div>',
                    unsafe_allow_html=True)
        st.markdown(fix_markdown(text))
