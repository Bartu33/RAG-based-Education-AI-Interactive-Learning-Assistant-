"""Lecture video player that reports where the student is (rag_ui/components/video_player).

The value is the player's latest event: {"event", "n", "t", "playing", "rate", "duration"}.
`moment()` turns it into the current second, counting the time played since that event.
"""
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

_player = components.declare_component(
    "video_player", path=str(Path(__file__).resolve().parent.parent / "components" / "video_player"))


def player(url: str, key: str, mode: str = "main", start: float = 0, seek: Optional[dict] = None,
           label: str = "", max_height: int = 560) -> Optional[dict]:
    """mode: "main" | "mini". seek: {"t": seconds, "n": counter}; a new n moves the video to t.
    Returns the newest event once (None when nothing new happened)."""
    value = _player(url=url, mode=mode, start=start, seek=seek, label=label, max_height=max_height,
                    key=key, default=None)
    seen = st.session_state.setdefault("vp_seen", {})
    if not value or seen.get(key) == value.get("n"):
        return None
    seen[key] = value["n"]
    st.session_state[f"vp_state_{key}"] = {**value, "received": time.time()}
    return value


def moment(key: str) -> float:
    """Second of the video the student is at now (0 before the first event)."""
    s = st.session_state.get(f"vp_state_{key}")
    if not s:
        return 0.0
    t = float(s["t"])
    if s["playing"]:
        t += (time.time() - s["received"]) * float(s.get("rate") or 1)
    return min(t, float(s["duration"])) if s.get("duration") else t
