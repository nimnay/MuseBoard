from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VisualCluster:
    cluster_id: str
    member_pin_ids: list[str]  # ordered most → least representative
    theme_label: str
    dominant_palette: list[str]
    representative_tags: list[str]
    cohesion: float = 0.0  # mean cosine similarity of members to the centroid


@dataclass
class MarketingStrategy:
    board_url: str
    cluster_id: str
    brand_guidelines: str
    palette_recommendations: list[dict]
    content_calendar: list[dict]
    generated_at: str
    search_keywords: list[str] = field(default_factory=list)
