"""Presentation-only helpers: brand assets (logo) and small reusable UI blocks.

Nothing here touches session_state, the backend client or any callback. The logo is read from
rag_ui/assets with an absolute path (pathlib), so it loads the same way on every rerun and from any
working directory, and it is embedded as a base64 data URI (cached until the file changes).
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Optional

import streamlit as st

ASSETS_DIR: Path = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH: Path = ASSETS_DIR / "logo.png"          # 256 px, transparent margins trimmed
LOGO_SM_PATH: Path = ASSETS_DIR / "logo_sm.png"    # 96 px: the small logos (sidebar, footer), a lighter rerun
FAVICON_PATH: Path = ASSETS_DIR / "favicon.png"    # 64 px
FALLBACK_ICON = ":material/school:"

APP_NAME = "Learning Assistant"
APP_TAGLINE = "Hybrid RAG · Privacy-first"


@st.cache_data(show_spinner=False)
def _data_uri(path: str, mtime: float) -> str:
    """The file as a data URI. `mtime` is only part of the cache key (a new file is read again)."""
    mime = "image/png" if path.lower().endswith(".png") else "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def asset_uri(path: Path) -> str:
    """A data URI for an asset, or "" when the file is missing (callers then show no image)."""
    if not path.is_file():
        return ""
    return _data_uri(str(path), path.stat().st_mtime)


def page_icon() -> str:
    """For st.set_page_config(page_icon=...): the logo file, or a Material icon if it is missing.
    Plain path check, no Streamlit call, so it is safe before set_page_config."""
    for p in (FAVICON_PATH, LOGO_PATH):
        if p.is_file():
            return str(p)
    return FALLBACK_ICON


def logo_html(size: int = 40, large: bool = False, alt: str = APP_NAME) -> str:
    """The logo as an <img> framed like a desktop app icon (rounded corners, soft shadow)."""
    small = size <= 48 and LOGO_SM_PATH.is_file()   # 96 px is sharp up to 48 px on a 2x screen
    uri = asset_uri(LOGO_SM_PATH if small else LOGO_PATH)
    if not uri:
        return ""
    cls = "brand-logo lg" if large else "brand-logo"
    return (f'<img class="{cls}" src="{uri}" width="{size}" height="{size}" '
            f'alt="{html.escape(alt)}" draggable="false">')


def brand_lockup_html(size: int = 40, name: str = APP_NAME, sub: Optional[str] = APP_TAGLINE) -> str:
    """Logo + app name (+ tagline) side by side."""
    sub_html = ""
    if sub:
        parts = [html.escape(s.strip()) for s in sub.split("·")]
        sub_html = f'<div class="brand-sub">{"<span class=dot></span>".join(parts)}</div>'
    return (f'<div class="brand-lockup">{logo_html(size)}<div class="brand-text">'
            f'<div class="brand-name">{html.escape(name)}</div>{sub_html}</div></div>')


def sidebar_brand() -> None:
    """The brand lockup in the top-left corner of the sidebar (header row, beside the collapse
    button; positioned by .st-key-side_brand in styles.css). Call inside `with st.sidebar:`."""
    with st.container(key="side_brand"):
        st.markdown(brand_lockup_html(40), unsafe_allow_html=True)


def empty_state(title: str, desc: str = "", icon: Optional[str] = None, logo: bool = False) -> None:
    """A calm, centred placeholder for screens with nothing to show yet.
    `icon` is a Material Symbols name (e.g. "forum"); `logo=True` shows the app logo instead."""
    if logo:
        mark = logo_html(64, large=True)
    elif icon:
        mark = f'<div class="es-icon"><span class="mi">{html.escape(icon)}</span></div>'
    else:
        mark = ""
    st.markdown(f'<div class="empty-state">{mark}<div class="es-title">{html.escape(title)}</div>'
                f'<div class="es-desc">{desc}</div></div>', unsafe_allow_html=True)


def sidebar_logo() -> None:
    """Native st.logo: shown top-left of the page while the sidebar is collapsed (styles.css hides
    it inside the open sidebar, where sidebar_brand() is shown instead)."""
    if LOGO_PATH.is_file():
        st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH), size="large")


def account_html(student_no: str) -> str:
    """Avatar initials + student number for the bottom of the sidebar. `student_no` must be escaped."""
    return (f'<div class="side-user"><span class="av">{student_no[:2].upper()}</span>'
            f'<div class="who"><div class="no">{student_no}</div><div class="role">Student</div></div></div>')


def status_html(icon: str, title: str, desc: str = "", tone: str = "info") -> str:
    """A compact status row (sidebar): coloured icon, one-line title, small description.
    tone: ok | info | warn | error. `desc` may hold simple HTML (escape any user / backend text)."""
    tone = tone if tone in {"ok", "info", "warn", "error"} else "info"
    d = f'<div class="sr-desc">{desc}</div>' if desc else ""
    return (f'<div class="status-row {tone}"><span class="mi">{html.escape(icon)}</span>'
            f'<div><div class="sr-title">{html.escape(title)}</div>{d}</div></div>')


def page_header(title: str, kicker: Optional[str] = None, subtitle: Optional[str] = None,
                icon: Optional[str] = None) -> None:
    """The heading block every page starts with: small label (with an optional Material icon),
    the page title and one line on what the page is for."""
    k = ""
    if kicker:
        ic = f'<span class="mi">{html.escape(icon)}</span>' if icon else ""
        k = f'<div class="kicker">{ic}{html.escape(kicker)}</div>'
    sub = f'<p class="page-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<div class="page-head">{k}<h1>{html.escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


# --- RAG sources (Q&A answers) ---------------------------------------------------------------------
_TYPE_LABEL = {"pdf": "PDF", "pptx": "Slides", "docx": "Word", "video": "Video"}
_TYPE_ICON = {"pdf": "picture_as_pdf", "pptx": "slideshow", "docx": "description", "video": "smart_display"}


def _mmss(ms: int) -> str:
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def source_location(s: dict) -> str:
    """'page 4', 'slide 7', '12:30–13:10' or '' (same rule the Q&A page always used)."""
    if s.get("type") == "video" and s.get("start_ms") is not None:
        return f"{_mmss(s['start_ms'])}–{_mmss(s['end_ms'] or s['start_ms'])}"
    if s.get("page"):
        return f"{'slide' if s.get('type') == 'pptx' else 'page'} {s['page']}"
    return ""


def source_chips_html(sources: list[dict], limit: int = 3) -> str:
    """A row of small chips under an answer: [n] document · location. The best-ranked source gets
    the gold accent (the one 'featured' source of the answer)."""
    chips = []
    for i, s in enumerate(sources[:limit], 1):
        loc = source_location(s)
        icon = _TYPE_ICON.get(s.get("type"), "draft")
        chips.append(f'<span class="src-chip{" top" if i == 1 else ""}" title="{html.escape(str(s.get("document", "")))}">'
                     f'<span class="n">{i}</span><span class="mi">{icon}</span>'
                     f'<span class="doc">{html.escape(str(s.get("document", "")))}</span>'
                     + (f'<span class="loc">{html.escape(loc)}</span>' if loc else "") + "</span>")
    more = len(sources) - limit
    if more > 0:
        chips.append(f'<span class="src-more">+{more} more</span>')
    return f'<div class="src-chips">{"".join(chips)}</div>'


def source_card_html(n: int, s: dict, top: bool = False) -> str:
    """One retrieved passage in the Sources list: number, type, document, location, score, excerpt."""
    loc = source_location(s)
    t = s.get("type", "")
    meta = " · ".join(x for x in (loc, f"score {s['score']}" if s.get("score") is not None else "") if x)
    return (f'<div class="src-card{" top" if top else ""}"><div class="src-head">'
            f'<span class="n">{n}</span><span class="type"><span class="mi">{_TYPE_ICON.get(t, "draft")}</span>'
            f'{html.escape(_TYPE_LABEL.get(t, t.upper() or "File"))}</span>'
            f'<span class="doc">{html.escape(str(s.get("document", "")))}</span>'
            + (f'<span class="meta">{html.escape(meta)}</span>' if meta else "")
            + (f'<span class="best">Best match</span>' if top else "")
            + f'</div><div class="src-text">{html.escape(str(s.get("text", "")))}</div></div>')
