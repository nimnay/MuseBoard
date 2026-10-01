"""Thin LLM clients shared by the vision, labeling and synthesis stages.

Model IDs are env-overridable because hosted models get retired; defaults were
checked against the Gemini and Groq model lists in October 2026.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Protocol

from curator_agent.config import get_env
from curator_agent.resilience import retry_with_backoff

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_VISION_MODEL = "qwen/qwen3.8-27b"


class LLM(Protocol):
    name: str
    model_id: str

    def generate_json(self, prompt: str, image_jpeg: bytes | None = None) -> dict:
        ...

    def generate_text(self, prompt: str) -> str:
        ...


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # Some models wrap JSON in a fenced block despite instructions.
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


class GeminiLLM:
    name = "gemini"

    def __init__(self) -> None:
        from google import genai

        self.model_id = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._client = genai.Client(api_key=get_env("GEMINI_API_KEY"))

    @retry_with_backoff
    def generate_json(self, prompt: str, image_jpeg: bytes | None = None) -> dict:
        from google.genai import types

        contents: list = [prompt]
        if image_jpeg is not None:
            contents.insert(0, types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"))
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
        )
        return _parse_json(resp.text or "")

    @retry_with_backoff
    def generate_text(self, prompt: str) -> str:
        resp = self._client.models.generate_content(model=self.model_id, contents=prompt)
        return (resp.text or "").strip()


class GroqLLM:
    name = "groq"

    def __init__(self) -> None:
        from groq import Groq

        self.model_id = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_TEXT_MODEL)
        self._vision_model_id = os.environ.get("GROQ_VISION_MODEL", DEFAULT_GROQ_VISION_MODEL)
        self._client = Groq(api_key=get_env("GROQ_API_KEY"))

    @retry_with_backoff
    def generate_json(self, prompt: str, image_jpeg: bytes | None = None) -> dict:
        content: list | str = prompt
        model = self.model_id
        if image_jpeg is not None:
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            model = self._vision_model_id
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return _parse_json(resp.choices[0].message.content or "")

    @retry_with_backoff
    def generate_text(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(provider: str) -> LLM | None:
    """Return an LLM client, or None for the key-free local provider."""
    if provider == "gemini":
        return GeminiLLM()
    if provider == "groq":
        return GroqLLM()
    return None
