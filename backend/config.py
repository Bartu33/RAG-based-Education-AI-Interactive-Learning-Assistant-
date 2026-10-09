from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p