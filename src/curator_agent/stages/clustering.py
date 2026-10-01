"""Clustering stage: group pins into visual themes and describe each one.

* k is chosen by cosine silhouette score unless the user fixes it, so a board
  with two clear looks isn't forced into four clusters.
* Tags are ranked by *distinctiveness* — how much more common they are inside a
  cluster than across the board — so labels separate clusters instead of all
  repeating the board's most common tag.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from curator_agent.images import merge_palettes
from curator_agent.logging_config import get_logger
from curator_agent.models.cluster import VisualCluster
from curator_agent.models.pin import PinEmbedding, PinMetadata

MAX_AUTO_K = 8
# (distinctive tags, mood) -> label
Labeler = Callable[[list[str], str], str]


@dataclass
class ClusteringResult:
    clusters: list[VisualCluster]
    k: int
    silhouette: float | None
    k_scores: dict[int, float] = field(default_factory=dict)


def _kmeans(matrix: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)


def choose_k(matrix: np.ndarray, k_max: int = MAX_AUTO_K) -> tuple[int, dict[int, float]]:
    """Return the k in [2, k_max] with the best cosine silhouette score."""
    from sklearn.metrics import silhouette_score

    n = len(matrix)
    if n < 3:
        return 1, {}
    scores = {
        k: float(silhouette_score(matrix, _kmeans(matrix, k), metric="cosine"))
        for k in range(2, min(k_max, n - 1) + 1)
    }
    best = max(scores, key=lambda k: (scores[k], -k))  # ties → fewer clusters
    return best, scores


def _pin_tags(m: PinMetadata) -> set[str]:
    return {t.lower() for t in m.aesthetic_tags} | {m.style_category.lower(), m.mood.lower()} - {""}


def distinctive_tags(
    member_ids: list[str], meta_by_id: dict[str, PinMetadata], top: int = 5
) -> list[str]:
    """Rank tags by (share of cluster with tag) − (share of board with tag)."""
    board_counts: Counter = Counter()
    for m in meta_by_id.values():
        board_counts.update(_pin_tags(m))
    cluster_counts: Counter = Counter()
    for pid in member_ids:
        cluster_counts.update(_pin_tags(meta_by_id[pid]))

    n_board, n_cluster = len(meta_by_id), len(member_ids)

    def rank(tag: str) -> tuple[float, float, str]:
        in_share = cluster_counts[tag] / n_cluster
        lift = in_share - board_counts[tag] / n_board
        return -lift, -in_share, tag  # most distinctive first; ties → more common, then A-Z

    return sorted(cluster_counts, key=rank)[:top]


def default_label(tags: list[str], mood: str) -> str:
    return " · ".join(t.title() for t in tags[:2]) or mood.title() or "Untitled"


LABEL_PROMPT = (
    "These tags describe one visual theme on a Pinterest board, most distinctive first: {tags}. "
    "The dominant mood is {mood}. Name the theme in 2-5 evocative words, the way a Pinterest "
    "board title would read. Reply with the name only — no quotes or punctuation at the ends."
)


def llm_labeler(llm) -> Labeler:
    def label(tags: list[str], mood: str) -> str:
        text = llm.generate_text(LABEL_PROMPT.format(tags=", ".join(tags), mood=mood))
        return text.strip().strip("\"'*.").splitlines()[0][:60] if text else ""

    return label


def cluster_embeddings(
    embeddings: list[PinEmbedding],
    metadata_list: list[PinMetadata],
    n_clusters: int | None,
    run_id: str,
    labeler: Labeler | None = None,
) -> ClusteringResult:
    log = get_logger("Clustering", run_id)
    meta_by_id = {m.pin_id: m for m in metadata_list}
    pin_ids = [e.pin_id for e in embeddings]
    matrix = np.array([e.embedding for e in embeddings], dtype=np.float32)

    k_scores: dict[int, float] = {}
    if n_clusters is None:
        k, k_scores = choose_k(matrix)
    else:
        k = max(1, min(n_clusters, len(matrix)))
    labels = np.zeros(len(matrix), dtype=int) if k == 1 else _kmeans(matrix, k)

    silhouette = k_scores.get(k)
    if silhouette is None and 1 < k < len(matrix):
        from sklearn.metrics import silhouette_score
        silhouette = float(silhouette_score(matrix, labels, metric="cosine"))
    log.info("k=%d silhouette=%s scores=%s", k, silhouette, k_scores)

    groups = sorted(
        (np.flatnonzero(labels == label) for label in np.unique(labels)),
        key=len, reverse=True,
    )
    clusters: list[VisualCluster] = []
    for i, idx in enumerate(groups):
        centroid = matrix[idx].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        sims = matrix[idx] @ centroid
        member_ids = [pin_ids[j] for j in idx[np.argsort(-sims)]]

        tags = distinctive_tags(member_ids, meta_by_id)
        mood = Counter(meta_by_id[p].mood for p in member_ids).most_common(1)[0][0]
        label = default_label(tags, mood)
        if labeler is not None:
            try:
                label = labeler(tags, mood) or label
            except Exception as exc:  # noqa: BLE001 — a label is nice-to-have
                log.warning("Theme labeling failed for cluster %d: %s", i, exc)

        clusters.append(VisualCluster(
            cluster_id=f"cluster_{i}",
            member_pin_ids=member_ids,
            theme_label=label,
            dominant_palette=merge_palettes([meta_by_id[p].color_palette for p in member_ids]),
            representative_tags=tags,
            cohesion=round(float(sims.mean()), 4),
        ))
    return ClusteringResult(clusters=clusters, k=k, silhouette=silhouette, k_scores=k_scores)
