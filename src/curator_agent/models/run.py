from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageResult:
    status: str  # "success" | "failure" | "skipped"
    input_count: int
    output_count: int
    elapsed_ms: int
    error: str | None = None


@dataclass
class PipelineRun:
    run_id: str
    board_url: str
    notion_db_id: str | None
    started_at: str
    completed_at: str | None = None
    stages: dict[str, StageResult] = field(default_factory=dict)
    exit_code: int = 0
    source: str = ""
    provider: str = ""


@dataclass
class NotionRecord:
    notion_page_id: str
    cluster_id: str
    export_timestamp: str
    export_status: str  # "success" | "skipped" | "failed"
    error_message: str | None = None
