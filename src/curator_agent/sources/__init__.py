"""Pin sources: where a board's pins come from.

* ``rss``   — Pinterest's public board feed. No auth; ~25 most recent pins.
* ``api``   — Pinterest API v5. Needs a token; reads boards the token owner can access.
* ``local`` — a JSON file of pins, for offline demos and tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from curator_agent.models.pin import Pin

_BOARD_URL_RE = re.compile(
    r"^https?://(?:[a-z]{2}\.|www\.)?pinterest\.[a-z.]+/([^/?#]+)/([^/?#]+)/?(?:[?#].*)?$",
    re.IGNORECASE,
)
# First path segments that are Pinterest pages rather than usernames.
_RESERVED_SEGMENTS = {"pin", "search", "ideas", "today", "categories", "business", "_"}


class SourceError(RuntimeError):
    """A source could not produce pins for a board (bad URL, private board, ...)."""


@dataclass(frozen=True)
class BoardRef:
    username: str
    slug: str

    @property
    def board_id(self) -> str:
        """Filesystem-safe identifier used for cache and output paths."""
        return f"{self.username}_{self.slug}"

    @property
    def url(self) -> str:
        return f"https://www.pinterest.com/{self.username}/{self.slug}/"


def parse_board_url(url: str) -> BoardRef:
    m = _BOARD_URL_RE.match(url.strip())
    if not m or m.group(1).lower() in _RESERVED_SEGMENTS:
        raise ValueError(
            f"Invalid Pinterest board URL: {url!r}. "
            "Expected https://www.pinterest.com/<username>/<board>/"
        )
    return BoardRef(username=m.group(1), slug=m.group(2))


def board_ref_for_file(path: Path) -> BoardRef:
    return BoardRef(username="local", slug=path.stem)


class PinSource(Protocol):
    name: str

    def fetch(self, board: BoardRef, max_pins: int) -> list[Pin]:
        ...
