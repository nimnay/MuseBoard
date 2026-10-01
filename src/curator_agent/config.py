"""Environment lookups and provider resolution."""
from __future__ import annotations

import os

# Values copied verbatim from .env.example look like "your_gemini_api_key_here".
_PLACEHOLDER_PREFIX = "your_"

USER_AGENT = "MuseBoard/0.2 (+https://github.com/nimnay/MuseBoard)"

LLM_PROVIDERS = ("gemini", "groq")


def is_unset(value: str | None) -> bool:
    """True for None, blanks, and placeholders copied verbatim from .env.example."""
    value = (value or "").strip()
    return not value or value.lower().startswith(_PLACEHOLDER_PREFIX)


def get_env(name: str) -> str | None:
    value = os.environ.get(name)
    return None if is_unset(value) else value.strip()  # type: ignore[union-attr]


def resolve_provider(requested: str) -> str:
    """Pick the vision/LLM provider.

    ``auto`` prefers Gemini, then Groq, and falls back to the fully local CLIP
    tagger so the pipeline always runs without any API keys.
    """
    if requested != "auto":
        return requested
    if get_env("GEMINI_API_KEY"):
        return "gemini"
    if get_env("GROQ_API_KEY"):
        return "groq"
    return "local"


def required_env_for(provider: str, source: str, notion_db: str | None) -> list[str]:
    """Env vars that must be set for this combination of options."""
    required = []
    if provider == "gemini":
        required.append("GEMINI_API_KEY")
    elif provider == "groq":
        required.append("GROQ_API_KEY")
    if source == "api":
        required.append("PINTEREST_ACCESS_TOKEN")
    if notion_db:
        required.append("NOTION_API_KEY")
    return required
