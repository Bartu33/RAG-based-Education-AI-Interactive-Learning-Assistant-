"""Settings → General presets (rag_ui/core/state.py) and the request settings the backend honours."""
import sys
from pathlib import Path

from tests.test_beyond import _qa
from tests.test_study import DOC, store  # noqa: F401  (fixture)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag_ui"))
from core.state import DEFAULT_CFG, PRESETS, preset_of  # noqa: E402


def test_presets_cover_the_same_settings_and_defaults_are_medium():
    assert {tuple(sorted(p)) for p in PRESETS.values()} == {tuple(sorted(PRESETS["Medium"]))}
    assert preset_of(DEFAULT_CFG) == "Medium"
    assert preset_of({**DEFAULT_CFG, **PRESETS["Low"]}) == "Low"
    assert preset_of({**DEFAULT_CFG, "top_k_context": 7}) is None          # changed in Advanced
    low, high = PRESETS["Low"], PRESETS["High"]
    assert low["top_k_context"] < PRESETS["Medium"]["top_k_context"] < high["top_k_context"]


def test_grounding_check_can_be_switched_off_per_request(monkeypatch):
    qa, scored = _qa(monkeypatch, "Signals vary with time [1].")
    assert qa.answer_question("q", "C", cfg={"score_faithfulness": False})["faithfulness"] is None
    assert scored == []
    assert qa.answer_question("q", "C", cfg={})["faithfulness"] == 0.9    # config.yaml default: on


def test_study_buddy_excerpt_count_per_request(store):  # noqa: F811
    from backend.tasks import study
    two = study.select_chunks("CENG336", DOC, "ask", "timers?", store=store, top_context=2)["chunks"]
    default = study.select_chunks("CENG336", DOC, "ask", "timers?", store=store)["chunks"]
    assert len(two) == 2 and len(default) == study.DEFAULTS["top_context"]
