import json
import logging
import os
import re
from typing import Optional

from backend.config import get_config

log = logging.getLogger(__name__)
_CACHE: dict = {}


class LLMProvider:
    """Groq, OpenAI and Ollama all expose an OpenAI-compatible API, so one class covers them."""

    def __init__(self, name: str):
        cfg = get_config()["llm"]
        if name not in cfg["models"]:
            raise ValueError(f"Unknown LLM provider: {name}")
        self.name = name
        self.model = cfg["models"][name]
        self.is_local = name == "ollama"
        key = "ollama" if self.is_local else os.getenv(f"{name.upper()}_API_KEY")
        if not key:
            raise RuntimeError(f"Missing {name.upper()}_API_KEY (set it in .env)")
        from openai import OpenAI
        self._client = OpenAI(base_url=cfg["base_urls"][name], api_key=key, timeout=60.0)

    def chat(self, messages: list[dict], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        cfg = get_config()["llm"]
        kwargs = dict(model=self.model, messages=messages,
                      temperature=cfg["temperature"] if temperature is None else temperature,
                      max_tokens=max_tokens or cfg["max_tokens"])
        if json_mode:
            try:
                resp = self._client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"})
            except Exception:  # some servers do not support json mode
                resp = self._client.chat.completions.create(**kwargs)
        else:
            resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def resolve_provider_name(label: Optional[str]) -> Optional[str]:
    """Map a UI label such as 'Ollama (Mistral) - Privacy mode' to a provider key."""
    if not label:
        return None
    low = label.lower()
    if "ollama" in low:
        return "ollama"
    if "openai" in low:
        return "openai"
    return "groq"


def _default_name(name: Optional[str]) -> str:
    return (name or os.getenv("LLM_PROVIDER") or get_config()["llm"]["provider"]).lower()


def get_provider(name: Optional[str] = None) -> LLMProvider:
    name = _default_name(name)
    if name not in _CACHE:
        _CACHE[name] = LLMProvider(name)
    return _CACHE[name]


def chat_with_fallback(messages: list[dict], provider: Optional[str] = None, **kwargs):
    """Returns (text, provider_used). Local (privacy) mode never falls back to a cloud API."""
    primary = _default_name(provider)
    order = [primary]
    if primary != "ollama":
        order += [n for n in get_config()["llm"].get("fallback", []) if n != primary]
    last: Exception | None = None
    for name in order:
        try:
            return get_provider(name).chat(messages, **kwargs), name
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM provider %s failed: %s", name, exc)
            last = exc
    raise RuntimeError(f"All LLM providers failed: {last}")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (handles ``` fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(text[start:end + 1])