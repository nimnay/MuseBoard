"""Synthesis stage: turn each visual cluster into a content strategy (LLM only)."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import jsonschema

from curator_agent.llm import LLM
from curator_agent.logging_config import get_logger
from curator_agent.models.cluster import MarketingStrategy, VisualCluster
from curator_agent.models.pin import Pin

STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["brand_guidelines", "palette_recommendations", "content_calendar", "search_keywords"],
    "properties": {
        "brand_guidelines": {"type": "string", "minLength": 1},
        "palette_recommendations": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["name", "hex", "usage"],
                "properties": {"name": {"type": "string"}, "hex": {"type": "string"}, "usage": {"type": "string"}},
            },
        },
        "content_calendar": {
            "type": "array", "minItems": 3,
            "items": {
                "type": "object", "required": ["theme", "description"],
                "properties": {"theme": {"type": "string"}, "description": {"type": "string"}},
            },
        },
        "search_keywords": {"type": "array", "minItems": 3, "items": {"type": "string"}},
    },
}

SYNTHESIS_PROMPT = """You are a creative strategist helping a creator turn a Pinterest mood board into a content plan.

One visual theme from their board:
{cluster_json}

Return ONLY a JSON object with:
- brand_guidelines: 2-4 sentences of concrete brand direction for this theme
- palette_recommendations: array of {{"name", "hex", "usage"}} built from the dominant palette (3-5 items)
- content_calendar: array of 4 {{"theme", "description"}} weekly post ideas that fit this look
- search_keywords: 5-8 phrases people would actually type into Pinterest search to find this look
"""


def synthesize_strategy(llm: LLM, cluster: VisualCluster, pins_by_id: dict[str, Pin], board_url: str) -> MarketingStrategy:
    captions = [
        pins_by_id[p].title for p in cluster.member_pin_ids[:6]
        if p in pins_by_id and pins_by_id[p].title
    ]
    cluster_json = json.dumps({
        "theme_label": cluster.theme_label,
        "distinctive_tags": cluster.representative_tags,
        "dominant_palette": cluster.dominant_palette,
        "pin_count": len(cluster.member_pin_ids),
        "example_pin_captions": [c[:160] for c in captions],
    }, indent=2)
    data = llm.generate_json(SYNTHESIS_PROMPT.format(cluster_json=cluster_json))
    jsonschema.validate(data, STRATEGY_SCHEMA)
    return MarketingStrategy(
        board_url=board_url,
        cluster_id=cluster.cluster_id,
        brand_guidelines=data["brand_guidelines"],
        palette_recommendations=data["palette_recommendations"],
        content_calendar=data["content_calendar"],
        search_keywords=data["search_keywords"],
        generated_at=datetime.now(UTC).isoformat(),
    )


def synthesize_all(
    llm: LLM,
    clusters: list[VisualCluster],
    pins: list[Pin],
    board_url: str,
    run_id: str,
) -> list[MarketingStrategy]:
    """One strategy per cluster; a failed cluster is skipped, not fatal."""
    log = get_logger("Synthesis", run_id)
    pins_by_id = {p.id: p for p in pins}
    strategies = []
    for cluster in clusters:
        try:
            strategies.append(synthesize_strategy(llm, cluster, pins_by_id, board_url))
        except Exception as exc:  # noqa: BLE001
            log.warning("Synthesis failed for %s: %s — skipping", cluster.cluster_id, exc)
    if clusters and not strategies:
        raise RuntimeError("Synthesis failed for every cluster; see logs.")
    return strategies
