"""CLIP image/text encoder — one shared embedding space for pixels and words.

Image and text vectors are L2-normalised, so a dot product is cosine similarity.
The model loads lazily and is reused across stages within a run.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"
_BATCH_SIZE = 16


class Encoder(Protocol):
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class ClipEncoder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self._model_name).eval()
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
        return self._model, self._processor

    @staticmethod
    def _as_array(features) -> np.ndarray:
        # transformers 5 returns a ModelOutput; 4.x returned the tensor itself.
        tensor = getattr(features, "pooler_output", features)
        return tensor.detach().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        model, processor = self._load()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(images), _BATCH_SIZE):
                inputs = processor(images=images[start:start + _BATCH_SIZE], return_tensors="pt")
                chunks.append(self._as_array(model.get_image_features(**inputs)))
        return _normalize(np.concatenate(chunks)) if chunks else np.zeros((0, 512), np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        model, processor = self._load()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), _BATCH_SIZE):
                # CLIP's text tower is limited to 77 tokens.
                inputs = processor(
                    text=texts[start:start + _BATCH_SIZE],
                    return_tensors="pt", padding=True, truncation=True, max_length=77,
                )
                chunks.append(self._as_array(model.get_text_features(**inputs)))
        return _normalize(np.concatenate(chunks)) if chunks else np.zeros((0, 512), np.float32)
