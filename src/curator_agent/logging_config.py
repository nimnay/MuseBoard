"""Structured JSON-lines logs per run, plus concise console output."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

LOG_DIR = Path("logs")


class JsonLinesFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "stage": record.name.rsplit(".", 1)[-1],
            "message": record.getMessage(),
        })


def get_logger(stage_name: str, run_id: str) -> logging.Logger:
    """Logger writing everything (INFO+) to logs/{run_id}.jsonl and LOG_LEVEL+ to stderr."""
    log = logging.getLogger(f"curator_agent.{stage_name}")
    if getattr(log, "_run_id", None) == run_id:
        return log

    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)

    console_level = logging.getLevelNamesMapping().get(
        os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
    )
    log.setLevel(min(logging.INFO, console_level))

    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f"{run_id}.jsonl", encoding="utf-8")
    fh.setFormatter(JsonLinesFormatter())
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    sh.setLevel(console_level)
    log.addHandler(sh)

    log.propagate = False
    log._run_id = run_id  # type: ignore[attr-defined]
    return log
