"""Pipeline orchestrator.

Ingestion → Images → Vision → Embedding → Clustering → Synthesis → Report → Export

Each stage is timed, recorded on the PipelineRun (persisted to cache/runs.sqlite),
and printed as one status line. A failed stage stops the run with a non-zero exit
code; results produced before an Export failure are already on disk.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import numpy as np

from curator_agent.cache.store import CacheStore
from curator_agent.clip import ClipEncoder, Encoder
from curator_agent.images import load_images
from curator_agent.llm import LLM, get_llm
from curator_agent.logging_config import get_logger
from curator_agent.models.run import PipelineRun, StageResult
from curator_agent.report import build_results, write_outputs
from curator_agent.sources import BoardRef, PinSource
from curator_agent.stages.clustering import cluster_embeddings, llm_labeler
from curator_agent.stages.embedding import (
    IMAGE_WEIGHT_LOCAL,
    IMAGE_WEIGHT_WITH_LLM,
    generate_embeddings,
    save_embeddings,
)
from curator_agent.stages.export import export_to_notion
from curator_agent.stages.ingestion import ingest_pins
from curator_agent.stages.synthesis import synthesize_all
from curator_agent.stages.vision import extract_vision_metadata

EXIT_STAGE_FAILED = 3
EXIT_EXPORT_FAILED = 4

T = TypeVar("T")


@dataclass
class RunConfig:
    board: BoardRef
    source: PinSource
    provider: str  # "gemini" | "groq" | "local"
    n_clusters: int | None = None  # None → choose by silhouette
    max_pins: int = 50
    notion_db_id: str | None = None
    out_dir: Path | None = None
    no_cache: bool = False


class StageFailed(Exception):
    def __init__(self, stage: str, exit_code: int) -> None:
        super().__init__(stage)
        self.stage = stage
        self.exit_code = exit_code


class Pipeline:
    def __init__(
        self,
        run_id: str,
        encoder: Encoder | None = None,
        llm_factory: Callable[[str], LLM | None] = get_llm,
    ) -> None:
        self.run_id = run_id
        self._encoder = encoder or ClipEncoder()
        self._llm_factory = llm_factory
        self._store = CacheStore()
        self._log = get_logger("Pipeline", run_id)
        self._run: PipelineRun | None = None

    def _record(self, name: str, result: StageResult) -> None:
        assert self._run is not None
        self._run.stages[name] = result
        mark = {"success": "✓", "failure": "✗", "skipped": "–"}[result.status]
        line = (
            f"[{name:<10}] {mark}  in={result.input_count:<3} out={result.output_count:<3} "
            f"{result.elapsed_ms / 1000:6.1f}s"
        )
        if result.error:
            line += f"  {result.error}"
        print(line, flush=True)

    def _stage(
        self,
        name: str,
        input_count: int,
        fn: Callable[[], T],
        count: Callable[[T], int] = len,  # type: ignore[assignment]
        exit_code: int = EXIT_STAGE_FAILED,
    ) -> T:
        t0 = time.monotonic()
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001
            elapsed = int((time.monotonic() - t0) * 1000)
            self._log.exception("%s stage failed", name)
            self._record(name, StageResult("failure", input_count, 0, elapsed, str(exc)))
            raise StageFailed(name, exit_code) from exc
        elapsed = int((time.monotonic() - t0) * 1000)
        self._record(name, StageResult("success", input_count, count(out), elapsed))
        return out

    def _skip(self, name: str, reason: str) -> None:
        self._record(name, StageResult("skipped", 0, 0, 0, reason))

    def run(self, cfg: RunConfig) -> int:
        self._run = run = PipelineRun(
            run_id=self.run_id,
            board_url=cfg.board.url,
            notion_db_id=cfg.notion_db_id,
            started_at=datetime.now(UTC).isoformat(),
            source=cfg.source.name,
            provider=cfg.provider,
        )
        try:
            run.exit_code = self._execute(cfg)
        except StageFailed as exc:
            print(f"\nFailed at {exc.stage}. Details: logs/{self.run_id}.jsonl", file=sys.stderr)
            run.exit_code = exc.exit_code
        run.completed_at = datetime.now(UTC).isoformat()
        self._store.save_run(run)
        return run.exit_code

    def _execute(self, cfg: RunConfig) -> int:
        llm = self._llm_factory(cfg.provider)

        pins = self._stage("Ingestion", 0, lambda: ingest_pins(
            cfg.board, cfg.source, cfg.max_pins, cfg.no_cache, self.run_id))

        def download() -> dict:
            images = load_images(pins, get_logger("Images", self.run_id))
            if not images:
                raise RuntimeError("None of the pin images could be downloaded.")
            return images

        images = self._stage("Images", len(pins), download)

        def encode() -> dict[str, np.ndarray]:
            ids = list(images)
            return dict(zip(ids, self._encoder.encode_images([images[i] for i in ids])))

        # Includes the one-time torch/CLIP load, which dominates on a cold start.
        image_vecs = self._stage("CLIP", len(images), encode)

        metadata = self._stage("Vision", len(images), lambda: extract_vision_metadata(
            pins, images, image_vecs, llm, self._encoder, self.run_id, cfg.no_cache))

        image_weight = IMAGE_WEIGHT_LOCAL if llm is None else IMAGE_WEIGHT_WITH_LLM
        embeddings = self._stage("Embedding", len(metadata), lambda: generate_embeddings(
            metadata, image_vecs, self._encoder, image_weight, self.run_id))
        try:
            save_embeddings(embeddings, cfg.board.board_id)
        except Exception as exc:  # noqa: BLE001 — the index is a by-product, not an output
            self._log.warning("Could not save FAISS index: %s", exc)

        clustering = self._stage("Clustering", len(embeddings), lambda: cluster_embeddings(
            embeddings, metadata, cfg.n_clusters, self.run_id,
            labeler=llm_labeler(llm) if llm else None,
        ), count=lambda r: len(r.clusters))

        strategies = []
        if llm is None:
            self._skip("Synthesis", "local provider — add an LLM key for strategies")
        else:
            strategies = self._stage("Synthesis", len(clustering.clusters), lambda: synthesize_all(
                llm, clustering.clusters, pins, cfg.board.url, self.run_id))

        results = build_results(
            board=cfg.board, source=cfg.source.name, provider=cfg.provider,
            model=getattr(llm, "model_id", None), run_id=self.run_id,
            pins=pins, metadata=metadata, clusters=clustering.clusters, strategies=strategies,
            k=clustering.k, silhouette=clustering.silhouette, k_scores=clustering.k_scores,
        )
        out_dir = cfg.out_dir or Path("output") / cfg.board.board_id
        json_path, html_path = self._stage(
            "Report", len(clustering.clusters), lambda: write_outputs(results, out_dir))

        if cfg.notion_db_id and strategies:
            self._stage("Export", len(strategies), lambda: export_to_notion(
                strategies, clustering.clusters, cfg.notion_db_id, self.run_id,
            ), count=lambda recs: sum(r.export_status == "success" for r in recs),
                exit_code=EXIT_EXPORT_FAILED)
        elif cfg.notion_db_id:
            self._skip("Export", "no strategies to export")

        sil = clustering.silhouette
        print(
            f"\n{len(clustering.clusters)} themes from {len(metadata)} pins"
            f"{'' if sil is None else f' (silhouette {sil:.3f})'}."
            f"\nReport:  {html_path.resolve()}\nResults: {json_path.resolve()}"
        )
        return 0
