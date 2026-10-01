"""Shared fixtures. Tests run fully offline: no CLIP download, no API keys."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from curator_agent import resilience

DIM = 512


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """cache/, logs/ and output/ are cwd-relative — keep each test's in its own dir."""
    monkeypatch.chdir(tmp_path)
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "PINTEREST_ACCESS_TOKEN", "NOTION_API_KEY",
                "NOTION_DATABASE_ID", "MUSEBOARD_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(resilience, "_sleep", lambda _s: None)
    return tmp_path


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class FakeEncoder:
    """Stands in for CLIP: images embed by mean color, texts by a stable hash."""

    def encode_images(self, images):
        rows = []
        for img in images:
            mean = np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
            v = np.full(DIM, 0.01, dtype=np.float32)
            v[:3] = mean
            rows.append(v)
        return _unit(np.array(rows))

    def encode_texts(self, texts):
        rows = [
            np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=DIM)
            for t in texts
        ]
        return _unit(np.array(rows, dtype=np.float32))


@pytest.fixture
def encoder():
    return FakeEncoder()


VISION_RESPONSE = {
    "primary_objects": ["mug"],
    "color_palette": ["#FFFFFF"],
    "aesthetic_tags": ["Cozy", "Minimalist"],
    "mood": "Serene",
    "style_category": "Product Photography",
    "composition_notes": "Centered subject on a plain backdrop.",
    "detected_text": None,
    "font_style": None,
    "visual_quirks": [],
}

STRATEGY_RESPONSE = {
    "brand_guidelines": "Warm, slow, tactile.",
    "palette_recommendations": [{"name": "Clay", "hex": "#B4452F", "usage": "accents"}],
    "content_calendar": [{"theme": f"Week {i}", "description": "Post idea"} for i in range(1, 5)],
    "search_keywords": ["cozy mug", "slow mornings", "ceramic decor"],
}


class FakeLLM:
    name = "fake"
    model_id = "fake-1"

    def __init__(self, vision_response=None):
        self.vision_response = vision_response or VISION_RESPONSE
        self.calls = 0

    def generate_json(self, prompt, image_jpeg=None):
        self.calls += 1
        return dict(self.vision_response if image_jpeg is not None else STRATEGY_RESPONSE)

    def generate_text(self, prompt):
        return '"Slow Morning Ceramics"'


@pytest.fixture
def make_board(tmp_path):
    """Write solid-color PNGs plus a local pins JSON; return the JSON path."""

    def _make(colors: list[tuple[int, int, int]], name: str = "board") -> Path:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir(exist_ok=True)
        pins = []
        for i, color in enumerate(colors):
            path = img_dir / f"{name}_{i}.png"
            Image.new("RGB", (40, 40), color).save(path)
            pins.append({"id": f"p{i}", "image_url": f"imgs/{path.name}", "title": f"Pin {i}"})
        board = tmp_path / f"{name}.json"
        board.write_text(json.dumps({"pins": pins}), encoding="utf-8")
        return board

    return _make
