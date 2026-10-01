from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pin:
    id: str
    image_url: str
    board_id: str
    title: str | None = None
    description: str | None = None
    source_link: str | None = None


@dataclass
class PinMetadata:
    pin_id: str
    primary_objects: list[str]
    color_palette: list[str]
    aesthetic_tags: list[str]
    mood: str
    style_category: str
    composition_notes: str
    visual_quirks: list[str]
    schema_valid: bool
    schema_version: str
    font_style: str | None = None
    detected_text: str | None = None


@dataclass
class PinEmbedding:
    pin_id: str
    embedding: list[float]
    content_hash: str
