"""Image download/cache and pixel-based color palettes.

Every stage reads pins' images from here, so each image is downloaded once per
machine. Palettes come from k-means over actual pixels rather than from an
LLM's guess at hex codes, so they're deterministic and match the image.
"""
from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from curator_agent.config import USER_AGENT
from curator_agent.models.pin import Pin
from curator_agent.resilience import retry_with_backoff

IMAGE_CACHE_DIR = Path("cache") / "images"
_PALETTE_SAMPLE_SIZE = 64


@retry_with_backoff
def _download(url: str, session: requests.Session) -> bytes:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.content


def load_image_bytes(url_or_path: str, session: requests.Session | None = None) -> bytes:
    """Return image bytes for an https URL (disk-cached) or a local file path."""
    if not url_or_path.startswith(("http://", "https://")):
        return Path(url_or_path).read_bytes()

    cache_path = IMAGE_CACHE_DIR / (hashlib.sha1(url_or_path.encode()).hexdigest() + ".img")
    if cache_path.exists():
        return cache_path.read_bytes()
    data = _download(url_or_path, session or requests.Session())
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    except OSError:
        pass  # caching is an optimisation; never fail a run over it
    return data


def load_images(pins: list[Pin], log: logging.Logger, workers: int = 8) -> dict[str, Image.Image]:
    """Download (or read from cache) every pin image in parallel; skip any that fail."""
    session = requests.Session()

    def load(pin: Pin) -> Image.Image | None:
        try:
            return open_rgb(load_image_bytes(pin.image_url, session))
        except Exception as exc:  # noqa: BLE001 — one dead image shouldn't sink the board
            log.warning("Skipping pin %s — image unavailable: %s", pin.id, exc)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load, pins))
    return {pin.id: img for pin, img in zip(pins, loaded) if img is not None}


def open_rgb(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def to_jpeg(img: Image.Image, max_side: int = 768) -> bytes:
    """Re-encode as a bounded-size JPEG for LLM upload (pins may be WebP/PNG/huge)."""
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    copy.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(c)) for c in np.clip(rgb, 0, 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex_to_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


def _kmeans_colors(pixels: np.ndarray, k: int, weights: np.ndarray | None = None) -> list[str]:
    from sklearn.cluster import KMeans

    k = min(k, len(np.unique(pixels, axis=0)))
    if k == 0:
        return []
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels, sample_weight=weights)
    shares = np.bincount(km.labels_, weights=weights, minlength=k)
    order = np.argsort(-shares)
    return [_hex(km.cluster_centers_[i]) for i in order]


def extract_palette(img: Image.Image, k: int = 5) -> list[str]:
    """Dominant colors of an image as hex codes, most prevalent first."""
    small = img.copy()
    # Nearest-neighbour keeps real pixel colors; antialiasing would invent blends.
    small.thumbnail((_PALETTE_SAMPLE_SIZE, _PALETTE_SAMPLE_SIZE), resample=Image.Resampling.NEAREST)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3)
    return _kmeans_colors(pixels, k)


def merge_palettes(palettes: list[list[str]], k: int = 5) -> list[str]:
    """Combine several images' palettes into one, weighting each color by its rank."""
    colors, weights = [], []
    for palette in palettes:
        for rank, code in enumerate(palette):
            colors.append(_hex_to_rgb(code))
            weights.append(1.0 / (rank + 1))
    if not colors:
        return []
    return _kmeans_colors(np.array(colors, dtype=np.float32), k, np.array(weights))
