"""Local JSON source, for offline demos and tests.

Accepts either a list of pins or ``{"pins": [...]}``. Each pin needs ``id`` and
``image_url``; ``image_url`` may be an https URL or a path relative to the file.
"""
from __future__ import annotations

import json
from pathlib import Path

from curator_agent.models.pin import Pin
from curator_agent.sources import BoardRef, SourceError


class LocalSource:
    name = "local"

    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch(self, board: BoardRef, max_pins: int) -> list[Pin]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceError(f"Could not read pins from {self._path}: {exc}") from exc

        items = data.get("pins", []) if isinstance(data, dict) else data
        pins = []
        for item in items[:max_pins]:
            image_url = str(item.get("image_url", ""))
            if image_url and not image_url.startswith(("http://", "https://")):
                image_url = str((self._path.parent / image_url).resolve())
            pins.append(Pin(
                id=str(item.get("id", "")),
                image_url=image_url,
                board_id=board.board_id,
                title=item.get("title"),
                description=item.get("description"),
                source_link=item.get("source_link"),
            ))
        return pins
