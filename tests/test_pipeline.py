"""End-to-end runs with a local board, fake CLIP and (optionally) a fake LLM."""
from __future__ import annotations

import json
from pathlib import Path

from curator_agent.cache.store import CacheStore
from curator_agent.pipeline import EXIT_STAGE_FAILED, Pipeline, RunConfig
from curator_agent.sources import board_ref_for_file
from curator_agent.sources.local import LocalSource
from tests.conftest import FakeLLM

REDS = [(220, 20, 20), (200, 30, 25), (235, 10, 40), (210, 40, 30)]
BLUES = [(20, 30, 220), (30, 20, 200), (10, 50, 230), (40, 40, 210)]


def _config(board_path: Path, provider: str = "local", **kw) -> RunConfig:
    return RunConfig(board=board_ref_for_file(board_path), source=LocalSource(board_path), provider=provider, **kw)


def test_local_run_produces_report_and_results(make_board, encoder):
    board = make_board(REDS + BLUES)
    code = Pipeline("run1", encoder=encoder).run(_config(board))

    assert code == 0
    out = Path("output") / "local_board"
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["metrics"]["k"] == 2
    groups = [{p["id"] for p in c["pins"]} for c in results["clusters"]]
    assert {"p0", "p1", "p2", "p3"} in groups and {"p4", "p5", "p6", "p7"} in groups
    assert all(c["strategy"] is None for c in results["clusters"])
    assert "<!doctype html>" in (out / "report.html").read_text(encoding="utf-8")

    run = CacheStore().load_run("run1")
    assert run.exit_code == 0 and run.stages["Synthesis"].status == "skipped"


def test_llm_run_adds_labels_and_strategies(make_board, encoder):
    board = make_board(REDS + BLUES)
    llm = FakeLLM()
    code = Pipeline("run2", encoder=encoder, llm_factory=lambda _p: llm).run(
        _config(board, provider="gemini", n_clusters=2))

    assert code == 0
    results = json.loads((Path("output") / "local_board" / "results.json").read_text(encoding="utf-8"))
    assert results["run"]["model"] == "fake-1"
    for cluster in results["clusters"]:
        assert cluster["theme_label"] == "Slow Morning Ceramics"
        assert cluster["strategy"]["search_keywords"] == ["cozy mug", "slow mornings", "ceramic decor"]


def test_missing_images_fail_the_run_with_stage_exit_code(tmp_path, encoder):
    board = tmp_path / "broken.json"
    board.write_text(json.dumps([{"id": "1", "image_url": "imgs/missing.png"}]))
    code = Pipeline("run3", encoder=encoder).run(_config(board))
    assert code == EXIT_STAGE_FAILED
    run = CacheStore().load_run("run3")
    assert run.stages["Ingestion"].status == "failure"  # not HTTPS and not an existing file
