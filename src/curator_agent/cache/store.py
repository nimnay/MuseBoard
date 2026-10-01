"""Local JSON + SQLite cache: pins per board, vision metadata per pin, run history.

Caching makes re-runs cheap — free-tier LLM quotas run out quickly when the same
board is re-analysed while iterating on clustering or the report.
"""
from __future__ import annotations

import errno
import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from curator_agent.models.pin import Pin, PinMetadata
from curator_agent.models.run import PipelineRun, StageResult

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
RUNS_DB = CACHE_DIR / "runs.sqlite"


def _write_json(path: Path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        logger.warning("Disk full — skipping cache write to %s", path)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt cache file %s: %s", path, exc)
        return None


def _ensure_db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(RUNS_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
    conn.commit()
    return conn


class CacheStore:
    def save_pins(self, board_id: str, source: str, pins: list[Pin]) -> None:
        _write_json(CACHE_DIR / board_id / f"pins.{source}.json", [asdict(p) for p in pins])

    def load_pins(self, board_id: str, source: str) -> Optional[list[Pin]]:
        data = _read_json(CACHE_DIR / board_id / f"pins.{source}.json")
        if data is None:
            return None
        try:
            return [Pin(**d) for d in data]
        except TypeError as exc:
            logger.warning("Ignoring stale pin cache for %s: %s", board_id, exc)
            return None

    def _metadata_path(self, cache_key: str, pin_id: str) -> Path:
        return CACHE_DIR / "vision" / cache_key / f"{pin_id}.json"

    def save_metadata(self, cache_key: str, meta: PinMetadata) -> None:
        _write_json(self._metadata_path(cache_key, meta.pin_id), asdict(meta))

    def load_metadata(self, cache_key: str, pin_id: str) -> Optional[PinMetadata]:
        data = _read_json(self._metadata_path(cache_key, pin_id))
        try:
            return PinMetadata(**data) if data is not None else None
        except TypeError:
            return None

    def save_run(self, run: PipelineRun) -> None:
        try:
            conn = _ensure_db()
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, data) VALUES (?, ?)",
                (run.run_id, json.dumps(asdict(run))),
            )
            conn.commit()
            conn.close()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Could not save run metadata for %s: %s", run.run_id, exc)

    def load_run(self, run_id: str) -> Optional[PipelineRun]:
        try:
            conn = _ensure_db()
            row = conn.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to load run %s: %s", run_id, exc)
            return None
        if row is None:
            return None
        d = json.loads(row[0])
        d["stages"] = {k: StageResult(**v) for k, v in d.get("stages", {}).items()}
        return PipelineRun(**d)
