from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

from curator_agent.sources import BoardRef, SourceError, parse_board_url
from curator_agent.sources.api import ApiSource, slugify
from curator_agent.sources.local import LocalSource
from curator_agent.sources.rss import RssSource, parse_feed, upgrade_image_url

BOARD = BoardRef("someone", "cozy-kitchens")

FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Cozy kitchens</title>
<item>
  <title>Warm oak shelves</title>
  <link>https://www.pinterest.com/pin/111/</link>
  <description>&lt;a href="https://www.pinterest.com/pin/111/"&gt;&lt;img src="https://i.pinimg.com/236x/aa/bb/cc/one.jpg"&gt;&lt;/a&gt;Warm oak shelves</description>
</item>
<item>
  <title></title>
  <link>https://www.pinterest.com/pin/222/</link>
  <description>&lt;img src="https://i.pinimg.com/236x/dd/ee/ff/two.jpg"&gt;Terracotta tiles &amp;amp; brass</description>
</item>
<item>
  <title>No image here</title>
  <link>https://www.pinterest.com/pin/333/</link>
  <description>just text</description>
</item>
</channel></rss>"""


@pytest.mark.parametrize("url, expected", [
    ("https://www.pinterest.com/someone/cozy-kitchens/", BoardRef("someone", "cozy-kitchens")),
    ("https://pinterest.com/someone/cozy-kitchens", BoardRef("someone", "cozy-kitchens")),
    ("https://uk.pinterest.com/someone/cozy-kitchens/?invite=1", BoardRef("someone", "cozy-kitchens")),
    ("https://www.pinterest.ca/someone/cozy-kitchens/", BoardRef("someone", "cozy-kitchens")),
])
def test_parse_board_url_accepts_board_urls(url, expected):
    assert parse_board_url(url) == expected


@pytest.mark.parametrize("url", [
    "https://www.pinterest.com/pin/12345/",
    "https://www.pinterest.com/someone/",
    "https://example.com/someone/board/",
    "not a url",
])
def test_parse_board_url_rejects_non_boards(url):
    with pytest.raises(ValueError):
        parse_board_url(url)


def test_board_ref_ids():
    assert BOARD.board_id == "someone_cozy-kitchens"
    assert BOARD.url == "https://www.pinterest.com/someone/cozy-kitchens/"


def test_parse_feed_extracts_pins_and_upgrades_images():
    pins = parse_feed(FEED, BOARD, max_pins=10)
    assert [p.id for p in pins] == ["111", "222"]  # item without an image is skipped
    assert pins[0].image_url == "https://i.pinimg.com/736x/aa/bb/cc/one.jpg"
    assert pins[0].title == "Warm oak shelves"
    assert pins[0].description is None  # identical to title → dropped
    assert pins[1].title is None
    assert pins[1].description == "Terracotta tiles & brass"
    assert all(p.board_id == BOARD.board_id for p in pins)


def test_parse_feed_respects_max_pins():
    assert len(parse_feed(FEED, BOARD, max_pins=1)) == 1


def test_parse_feed_rejects_non_xml():
    with pytest.raises(SourceError):
        parse_feed(b"<html>nope", BOARD, 10)


def test_upgrade_image_url_leaves_other_hosts_alone():
    assert upgrade_image_url("https://example.com/236x/a.jpg") == "https://example.com/236x/a.jpg"


def _response(status: int, body: bytes = b"", json_body=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = body
    resp.json.return_value = json_body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def test_rss_source_fetches_feed():
    session = MagicMock()
    session.get.return_value = _response(200, FEED)
    pins = RssSource(session).fetch(BOARD, 50)
    assert len(pins) == 2
    assert session.get.call_args.args[0] == "https://www.pinterest.com/someone/cozy-kitchens.rss"


def test_rss_source_404_is_a_friendly_error_without_retries():
    session = MagicMock()
    session.get.return_value = _response(404)
    with pytest.raises(SourceError, match="public"):
        RssSource(session).fetch(BOARD, 50)
    assert session.get.call_count == 1


def test_slugify_matches_pinterest_board_slugs():
    assert slugify("Cozy Kitchens!") == "cozy-kitchens"
    assert slugify("  Café & Brunch  ") == "caf-brunch"


def test_api_source_finds_board_by_slug_and_paginates():
    boards = {"items": [
        {"id": "9", "name": "Other", "owner": {"username": "someone"}},
        {"id": "42", "name": "Cozy Kitchens", "owner": {"username": "someone"}},
    ], "bookmark": None}
    page1 = {"items": [{"id": "a", "media": {"images": {"600x": {"url": "https://i/a.jpg"}}}}], "bookmark": "next"}
    page2 = {"items": [{"id": "b", "title": "B", "media": {"images": {"1200x": {"url": "https://i/b.jpg"}}}}], "bookmark": None}
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = [_response(200, json_body=j) for j in (boards, page1, page2)]

    pins = ApiSource("tok", session).fetch(BOARD, 50)

    assert session.headers["Authorization"] == "Bearer tok"
    assert [p.id for p in pins] == ["a", "b"]
    assert pins[1].image_url == "https://i/b.jpg"
    assert session.get.call_args_list[1].args[0].endswith("/boards/42/pins")
    assert session.get.call_args_list[2].kwargs["params"]["bookmark"] == "next"


def test_api_source_explains_boards_the_token_cannot_see():
    session = MagicMock()
    session.headers = {}
    session.get.return_value = _response(200, json_body={"items": [], "bookmark": None})
    with pytest.raises(SourceError, match="--source rss"):
        ApiSource("tok", session).fetch(BOARD, 50)


def test_local_source_resolves_relative_image_paths(tmp_path):
    board = tmp_path / "pins.json"
    board.write_text(json.dumps([
        {"id": "1", "image_url": "imgs/a.png", "title": "A"},
        {"id": "2", "image_url": "https://example.com/b.jpg"},
    ]))
    pins = LocalSource(board).fetch(BoardRef("local", "pins"), 10)
    assert pins[0].image_url == str((tmp_path / "imgs" / "a.png").resolve())
    assert pins[1].image_url == "https://example.com/b.jpg"


def test_local_source_reports_bad_json(tmp_path):
    board = tmp_path / "pins.json"
    board.write_text("{oops")
    with pytest.raises(SourceError):
        LocalSource(board).fetch(BoardRef("local", "pins"), 10)
