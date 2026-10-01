"""Ingestion stage: fetch pins from a source, validate, and cache them."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from curator_agent.cache.store import CacheStore
from curator_agent.logging_config import get_logger
from curator_agent.models.pin import Pin
from curator_agent.sources import BoardRef, PinSource, SourceError


def _validate_pin(pin: Pin) -> None:
    if not pin.id:
        raise ValueError("pin has an empty id")
    if Path(pin.image_url).is_file():
        return
    if urlparse(pin.image_url).scheme != "https":
        raise ValueError(f"pin {pin.id} image_url is not HTTPS or a local file: {pin.image_url!r}")


def ingest_pins(
    board: BoardRef,
    source: PinSource,
    max_pins: int,
    no_cache: bool,
    run_id: str,
) -> list[Pin]:
    log = get_logger("Ingestion", run_id)
    store = CacheStore()

    if not no_cache:
        cached = store.load_pins(board.board_id, source.name)
        if cached:
            log.info("Loaded %d pins from cache", len(cached))
            return cached[:max_pins]

    raw = source.fetch(board, max_pins)
    pins: list[Pin] = []
    seen: set[str] = set()
    for pin in raw:
        try:
            _validate_pin(pin)
        except ValueError as exc:
            log.warning("Skipping invalid pin: %s", exc)
            continue
        if pin.id not in seen:
            seen.add(pin.id)
            pins.append(pin)

    if not pins:
        raise SourceError(f"{source.name} source returned 0 valid pins for {board.url}")

    store.save_pins(board.board_id, source.name, pins)
    log.info("Fetched %d pins via %s", len(pins), source.name)
    return pins
