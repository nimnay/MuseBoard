"""Public board RSS feed source (no authentication required)."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

import requests

from curator_agent.config import USER_AGENT
from curator_agent.models.pin import Pin
from curator_agent.resilience import retry_with_backoff
from curator_agent.sources import BoardRef, SourceError

FEED_URL = "https://www.pinterest.com/{username}/{slug}.rss"

_PIN_ID_RE = re.compile(r"/pin/(\d+)")
_IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# Feed thumbnails are 236px wide; the same path is served at 736px.
_PINIMG_SIZE_RE = re.compile(r"(//i\.pinimg\.com/)\d+x/")


def upgrade_image_url(url: str) -> str:
    return _PINIMG_SIZE_RE.sub(r"\g<1>736x/", url)


def _strip_html(text: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())


def parse_feed(xml_bytes: bytes, board: BoardRef, max_pins: int) -> list[Pin]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise SourceError(f"Board feed is not valid RSS: {exc}") from exc

    pins: list[Pin] = []
    for item in root.iter("item"):
        if len(pins) >= max_pins:
            break
        link = item.findtext("link") or item.findtext("guid") or ""
        id_match = _PIN_ID_RE.search(link)
        description_html = item.findtext("description") or ""
        img_match = _IMG_SRC_RE.search(description_html)
        if not id_match or not img_match:
            continue

        title = _strip_html(item.findtext("title") or "") or None
        description: str | None = _strip_html(description_html) or None
        if description == title:
            description = None

        pins.append(Pin(
            id=id_match.group(1),
            image_url=upgrade_image_url(html.unescape(img_match.group(1))),
            board_id=board.board_id,
            title=title,
            description=description,
        ))
    return pins


@retry_with_backoff
def _download_feed(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.content


class RssSource:
    name = "rss"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, board: BoardRef, max_pins: int) -> list[Pin]:
        url = FEED_URL.format(username=board.username, slug=board.slug)
        try:
            content = _download_feed(self._session, url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise SourceError(
                    f"No public feed for {board.url} — check the URL and that the board is public."
                ) from exc
            raise
        return parse_feed(content, board, max_pins)
