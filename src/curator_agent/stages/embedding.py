"""Embedding stage: hybrid image + text vectors in CLIP's shared space.

Each pin's vector blends what the image *looks like* (CLIP image embedding) with
what it *means* (CLIP text embedding of its vision metadata). Pure image
embeddings over-weight color and layout; pure text embeddings inherit every
quirk of the LLM's wording. Blending in one space gets the benefit of both.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from curator_agent.clip import Encoder
from curator_agent.logging_config import get_logger
from curator_agent.models.pin import PinEmbedding, PinMetadata

# Share of the final vector that comes from pixels. With local tagging the text
# is itself derived from the image, so the image vector is used alone.
IMAGE_WEIGHT_WITH_LLM = 0.5
IMAGE_WEIGHT_LOCAL = 1.0


def compose_text(m: PinMetadata) -> str:
    parts = [
        m.mood,
        m.style_category,
        ", ".join(m.aesthetic_tags),
        ", ".join(m.primary_objects),
        ", ".join(m.visual_quirks),
        m.font_style or "",
    ]
    return ". ".join(p for p in parts if p)


def generate_embeddings(
    metadata_list: list[PinMetadata],
    image_vecs: dict[str, np.ndarray],
    encoder: Encoder,
    image_weight: float,
    run_id: str,
) -> list[PinEmbedding]:
    log = get_logger("Embedding", run_id)
    texts = [compose_text(m) for m in metadata_list]
    image_matrix = np.stack([image_vecs[m.pin_id] for m in metadata_list])
    if image_weight < 1.0:
        text_matrix = encoder.encode_texts(texts)
        blended = image_weight * image_matrix + (1 - image_weight) * text_matrix
    else:
        blended = image_matrix
    blended /= np.clip(np.linalg.norm(blended, axis=1, keepdims=True), 1e-12, None)

    embeddings = [
        PinEmbedding(
            pin_id=m.pin_id,
            embedding=vec.astype(np.float32).tolist(),
            content_hash=hashlib.sha256(f"{image_weight}|{text}".encode()).hexdigest(),
        )
        for m, vec, text in zip(metadata_list, blended, texts)
    ]
    log.info("Embedded %d pins (image weight %.2f)", len(embeddings), image_weight)
    return embeddings


def save_embeddings(embeddings: list[PinEmbedding], board_id: str) -> Path:
    """Persist a FAISS inner-product index (cosine, since vectors are unit-length)."""
    import faiss

    cache_dir = Path("cache") / board_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.array([e.embedding for e in embeddings], dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(cache_dir / "embeddings.index"))
    (cache_dir / "pin_ids.json").write_text(
        json.dumps([e.pin_id for e in embeddings], indent=2), encoding="utf-8"
    )
    return cache_dir
