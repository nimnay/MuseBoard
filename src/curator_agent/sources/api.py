"""Pinterest API v5 source.

v5 addresses boards by numeric id, not by ``username/slug``, and a token can
only list boards its owner has access to. So we list the owner's boards, match
the one whose name slugifies to the URL slug, then page through its pins.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

import requests

from curator_agent.models.pin import Pin
from curator_agent.resilience import retry_with_backoff
from curator_agent.sources import BoardRef, SourceError

API_BASE = "https://api.pinterest.com/v5"
_PAGE_SIZE = 100
_IMAGE_SIZES = ("1200x", "600x", "400x300", "150x150")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _best_image_url(item: dict) -> str:
    images = (item.get("media") or {}).get("images") or {}
    for size in _IMAGE_SIZES:
        if size in images:
            return images[size].get("url", "")
    return next(iter(images.values()), {}).get("url", "") if images else ""


class ApiSource:
    name = "api"

    def __init__(self, token: str, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"

    @retry_with_backoff
    def _get(self, path: str, params: dict) -> dict:
        resp = self._session.get(f"{API_BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, path: str, limit: int | None = None) -> Iterator[dict]:
        bookmark: str | None = None
        seen = 0
        while True:
            params: dict = {"page_size": _PAGE_SIZE}
            if bookmark:
                params["bookmark"] = bookmark
            data = self._get(path, params)
            for item in data.get("items", []):
                yield item
                seen += 1
                if limit is not None and seen >= limit:
                    return
            bookmark = data.get("bookmark")
            if not bookmark or not data.get("items"):
                return

    def _find_board_id(self, board: BoardRef) -> str:
        for item in self._paginate("/boards"):
            owner = (item.get("owner") or {}).get("username", "")
            if owner.lower() == board.username.lower() and slugify(item.get("name", "")) == board.slug.lower():
                return item["id"]
        raise SourceError(
            f"Board {board.url} isn't visible to this API token. The API source can only read "
            "boards the token's account owns or collaborates on — use --source rss for public boards."
        )

    def fetch(self, board: BoardRef, max_pins: int) -> list[Pin]:
        board_id = self._find_board_id(board)
        pins = []
        for item in self._paginate(f"/boards/{board_id}/pins", limit=max_pins):
            pins.append(Pin(
                id=item.get("id", ""),
                image_url=_best_image_url(item),
                board_id=board.board_id,
                title=item.get("title") or None,
                description=item.get("description") or None,
                source_link=item.get("link") or None,
            ))
        return pins
