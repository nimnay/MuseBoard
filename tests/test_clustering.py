from __future__ import annotations

import numpy as np

from curator_agent.models.pin import PinEmbedding, PinMetadata
from curator_agent.stages.clustering import choose_k, cluster_embeddings, distinctive_tags


def _meta(pin_id: str, tags: list[str], mood: str = "cozy", palette=None) -> PinMetadata:
    return PinMetadata(
        pin_id=pin_id, primary_objects=[], color_palette=palette or ["#FFFFFF"], aesthetic_tags=tags,
        mood=mood, style_category="photo", composition_notes="x", visual_quirks=[],
        schema_valid=True, schema_version="1.1.0",
    )


def _blobs(centers: int, per: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(centers):
        center = np.zeros(dim)
        center[c] = 1.0
        rows.extend(center + rng.normal(scale=0.05, size=dim) for _ in range(per))
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_choose_k_finds_the_true_number_of_groups():
    k, scores = choose_k(_blobs(centers=3, per=6))
    assert k == 3
    assert scores[3] == max(scores.values())


def test_choose_k_with_too_few_points():
    assert choose_k(_blobs(1, 2)) == (1, {})


def test_distinctive_tags_prefer_what_sets_a_cluster_apart():
    meta = {
        "a": _meta("a", ["cozy", "cottagecore"]),
        "b": _meta("b", ["cozy", "cottagecore"]),
        "c": _meta("c", ["cozy", "brutalist"]),
        "d": _meta("d", ["cozy", "brutalist"]),
    }
    tags = distinctive_tags(["a", "b"], meta)
    # "cozy" is on every pin, so it doesn't distinguish the cluster.
    assert tags[0] == "cottagecore"
    assert tags.index("cottagecore") < tags.index("cozy")


def test_cluster_embeddings_end_to_end():
    matrix = _blobs(centers=2, per=5)
    embeddings = [PinEmbedding(f"p{i}", row.tolist(), "h") for i, row in enumerate(matrix)]
    metadata = [
        _meta(f"p{i}", ["cottagecore"] if i < 5 else ["y2k"], palette=["#AA0000"] if i < 5 else ["#0000AA"])
        for i in range(10)
    ]
    result = cluster_embeddings(embeddings, metadata, None, "run")

    assert result.k == 2 and result.silhouette > 0.5
    groups = [set(c.member_pin_ids) for c in result.clusters]
    assert {f"p{i}" for i in range(5)} in groups
    for c in result.clusters:
        assert 0.9 < c.cohesion <= 1.0
        assert c.theme_label in {"Cottagecore · Cozy", "Y2K · Cozy"}
        assert len(c.dominant_palette) == 1


def test_fixed_k_and_labeler_failure_falls_back():
    matrix = _blobs(centers=2, per=3)
    embeddings = [PinEmbedding(f"p{i}", row.tolist(), "h") for i, row in enumerate(matrix)]
    metadata = [_meta(f"p{i}", ["boho"]) for i in range(6)]

    def broken_labeler(tags, mood):
        raise RuntimeError("quota")

    result = cluster_embeddings(embeddings, metadata, 1, "run", labeler=broken_labeler)
    assert result.k == 1 and result.silhouette is None
    assert result.clusters[0].theme_label == "Boho · Cozy"  # ties broken alphabetically
