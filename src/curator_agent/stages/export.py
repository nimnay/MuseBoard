"""Notion export stage: one database page per cluster (optional)."""
from __future__ import annotations

from datetime import UTC, datetime

from curator_agent.config import get_env
from curator_agent.logging_config import get_logger
from curator_agent.models.cluster import MarketingStrategy, VisualCluster
from curator_agent.models.run import NotionRecord
from curator_agent.resilience import retry_with_backoff

_RICH_TEXT_LIMIT = 2000


class ExportError(RuntimeError):
    """Export can't proceed at all (bad credentials, missing database)."""


def _text(content: str) -> list[dict]:
    return [{"text": {"content": content[:_RICH_TEXT_LIMIT]}}]


def build_payload(strategy: MarketingStrategy, cluster: VisualCluster, notion_db_id: str) -> dict:
    calendar = "; ".join(f"{i.get('theme', '')}: {i.get('description', '')}" for i in strategy.content_calendar)
    return {
        "parent": {"database_id": notion_db_id},
        "properties": {
            "Name": {"title": _text(cluster.theme_label)},
            "Cluster ID": {"rich_text": _text(cluster.cluster_id)},
            # Notion multi-select options can't contain commas.
            "Tags": {"multi_select": [{"name": t.replace(",", " ")[:100]} for t in cluster.representative_tags]},
            "Palette": {"rich_text": _text(", ".join(cluster.dominant_palette))},
            "Brand Guidelines": {"rich_text": _text(strategy.brand_guidelines)},
            "Content Calendar": {"rich_text": _text(calendar)},
            "Pin Count": {"number": len(cluster.member_pin_ids)},
            "Board URL": {"url": strategy.board_url},
            "Generated At": {"date": {"start": strategy.generated_at}},
        },
    }


def export_to_notion(
    strategies: list[MarketingStrategy],
    clusters: list[VisualCluster],
    notion_db_id: str,
    run_id: str,
) -> list[NotionRecord]:
    from notion_client import Client
    from notion_client.errors import APIResponseError

    log = get_logger("Export", run_id)
    notion = Client(auth=get_env("NOTION_API_KEY"))
    create_page = retry_with_backoff(notion.pages.create)
    cluster_by_id = {c.cluster_id: c for c in clusters}
    now = datetime.now(UTC).isoformat()

    records: list[NotionRecord] = []
    for strategy in strategies:
        cluster = cluster_by_id[strategy.cluster_id]
        try:
            page = create_page(**build_payload(strategy, cluster, notion_db_id))
            records.append(NotionRecord(page["id"], cluster.cluster_id, now, "success"))
        except APIResponseError as exc:
            if exc.status in (401, 403, 404):
                raise ExportError(
                    f"Notion rejected the export (HTTP {exc.status}). Check NOTION_API_KEY and that "
                    f"database {notion_db_id!r} is shared with your integration."
                ) from exc
            # 400 = this record doesn't fit the database schema; keep going.
            log.error("Notion export failed for %s (HTTP %s): %s", cluster.cluster_id, exc.status, exc)
            status = "skipped" if exc.status == 400 else "failed"
            records.append(NotionRecord("", cluster.cluster_id, now, status, str(exc)))
    return records
