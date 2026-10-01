"""Vision stage: turn each pin image into structured PinMetadata.

Two backends:

* LLM (Gemini or Groq): rich descriptions, validated against the versioned JSON
  schema in ``schemas/``. If fewer than 90% of pins validate, the stage fails
  rather than clustering on garbage.
* Local: CLIP zero-shot tagging against a curated aesthetic vocabulary. Coarser,
  but needs no API key and no network beyond the image download.

Either way ``color_palette`` comes from pixels (see images.py), not the model.
"""
from __future__ import annotations

import json
from importlib import resources

import jsonschema
import numpy as np
from PIL import Image

from curator_agent.cache.store import CacheStore
from curator_agent.clip import Encoder
from curator_agent.images import extract_palette, to_jpeg
from curator_agent.llm import LLM
from curator_agent.logging_config import get_logger
from curator_agent.models.pin import Pin, PinMetadata

SCHEMA: dict = json.loads(
    resources.files("curator_agent.schemas").joinpath("vision-schema.json").read_text(encoding="utf-8")
)
SCHEMA_VERSION: str = SCHEMA["version"]
VALIDITY_THRESHOLD = 0.90

VISION_PROMPT = """You are tagging a Pinterest pin for aesthetic analysis.
Pin caption (may be empty): {caption}

Return ONLY a JSON object with these fields:
- primary_objects: array of strings — main subjects visible
- color_palette: array of hex strings like "#A3B1C2" — dominant colors
- aesthetic_tags: array of 3-6 strings — aesthetic keywords a Pinner would search (e.g. "cottagecore", "quiet luxury")
- mood: string — one mood word (e.g. "serene", "energetic")
- style_category: string — visual style label (e.g. "minimalist", "bohemian")
- composition_notes: string — one sentence on composition and framing
- detected_text: string or null — visible text, null if none
- font_style: string or null — dominant typography style, null if no text
- visual_quirks: array of strings — techniques like "film grain", "flat lay"; empty if none
"""

# Vocabulary for local zero-shot tagging, biased toward terms people search on Pinterest.
AESTHETICS = [
    "cottagecore", "dark academia", "minimalist", "maximalist", "scandinavian", "japandi",
    "mid-century modern", "art deco", "bohemian", "coastal", "industrial", "brutalist", "y2k",
    "retro 70s", "vintage", "french country", "modern farmhouse", "grandmillennial",
    "quiet luxury", "streetwear", "old money", "whimsical", "gothic", "pastel",
    "neon cyberpunk", "earthy natural", "tropical", "parisian chic", "clean girl", "eclectic",
]
MOODS = [
    "cozy", "serene", "energetic", "romantic", "nostalgic", "playful",
    "luxurious", "moody", "fresh", "festive", "dreamy", "bold",
]
STYLES = [
    "product photography", "interior design photo", "fashion editorial", "food photography",
    "illustration", "graphic design poster", "typography quote", "street photography",
    "landscape photography", "flat lay", "close-up macro shot", "collage", "portrait photography",
]
SUBJECTS = [
    "food", "drinks", "a living room", "a bedroom", "a kitchen", "a bathroom", "an outfit", "shoes",
    "jewelry", "nails", "a hairstyle", "makeup", "flowers", "plants", "holiday decor",
    "a table setting", "architecture", "nature", "animals", "an art print", "furniture",
    "candles", "books", "gifts", "people",
]


class LocalTagger:
    """CLIP zero-shot tagger: scores each image against text prompts for every vocabulary term."""

    def __init__(self, encoder: Encoder) -> None:
        self._groups = {
            "aesthetic": (AESTHETICS, [f"a {t} aesthetic photo" for t in AESTHETICS]),
            "mood": (MOODS, [f"an image with a {t} mood" for t in MOODS]),
            "style": (STYLES, [f"a {t}" for t in STYLES]),
            "subject": (SUBJECTS, [f"a photo of {t}" for t in SUBJECTS]),
        }
        self._text_vecs = {
            name: encoder.encode_texts(prompts) for name, (_, prompts) in self._groups.items()
        }

    def _top(self, group: str, image_vec: np.ndarray, n: int) -> list[str]:
        terms = self._groups[group][0]
        scores = self._text_vecs[group] @ image_vec
        return [terms[i].removeprefix("a ").removeprefix("an ") for i in np.argsort(-scores)[:n]]

    def tag(self, pin_id: str, image_vec: np.ndarray, palette: list[str]) -> PinMetadata:
        style = self._top("style", image_vec, 1)[0]
        return PinMetadata(
            pin_id=pin_id,
            primary_objects=self._top("subject", image_vec, 2),
            color_palette=palette,
            aesthetic_tags=self._top("aesthetic", image_vec, 3),
            mood=self._top("mood", image_vec, 1)[0],
            style_category=style,
            composition_notes=f"{style} (local CLIP zero-shot tagging)",
            visual_quirks=[],
            schema_valid=True,
            schema_version=SCHEMA_VERSION,
        )


def validate_response(response: dict, pin_id: str, palette: list[str]) -> PinMetadata:
    """Validate an LLM response against the schema; raises jsonschema.ValidationError."""
    jsonschema.validate(response, SCHEMA)
    return PinMetadata(
        pin_id=pin_id,
        primary_objects=response["primary_objects"],
        color_palette=palette,
        aesthetic_tags=[t.lower() for t in response["aesthetic_tags"]],
        mood=response["mood"].lower(),
        style_category=response["style_category"].lower(),
        composition_notes=response["composition_notes"],
        font_style=response.get("font_style"),
        visual_quirks=response.get("visual_quirks", []),
        detected_text=response.get("detected_text"),
        schema_valid=True,
        schema_version=SCHEMA_VERSION,
    )


def _caption(pin: Pin) -> str:
    return " ".join(filter(None, [pin.title, pin.description]))[:500]


def extract_vision_metadata(
    pins: list[Pin],
    images: dict[str, Image.Image],
    image_vecs: dict[str, np.ndarray],
    llm: LLM | None,
    encoder: Encoder,
    run_id: str,
    no_cache: bool = False,
) -> list[PinMetadata]:
    log = get_logger("Vision", run_id)
    store = CacheStore()
    if llm is None:
        cache_key = "local-clip"
        tagger = LocalTagger(encoder)
    else:
        cache_key = f"{llm.name}-{llm.model_id}-v{SCHEMA_VERSION}".replace("/", "_")

    results: list[PinMetadata] = []
    attempted = 0
    for i, pin in enumerate(pins, 1):
        if pin.id not in images:
            continue
        attempted += 1
        cached = None if no_cache else store.load_metadata(cache_key, pin.id)
        if cached is not None:
            results.append(cached)
            continue

        palette = extract_palette(images[pin.id])
        if llm is None:
            meta = tagger.tag(pin.id, image_vecs[pin.id], palette)
        else:
            log.info("Vision %d/%d: pin %s", i, len(pins), pin.id)
            try:
                response = llm.generate_json(
                    VISION_PROMPT.format(caption=_caption(pin) or "(none)"),
                    image_jpeg=to_jpeg(images[pin.id]),
                )
                meta = validate_response(response, pin.id, palette)
            except jsonschema.ValidationError as exc:
                log.warning("Pin %s failed schema validation: %s", pin.id, exc.message)
                continue
            except Exception as exc:  # noqa: BLE001 — one bad pin shouldn't sink the board
                log.warning("Pin %s vision call failed: %s", pin.id, exc)
                continue
        store.save_metadata(cache_key, meta)
        results.append(meta)

    if attempted == 0:
        raise RuntimeError("No pin images could be loaded.")
    valid_ratio = len(results) / attempted
    if valid_ratio < VALIDITY_THRESHOLD:
        raise RuntimeError(
            f"Only {len(results)} of {attempted} pins ({valid_ratio:.0%}) produced valid metadata; "
            f"need ≥{VALIDITY_THRESHOLD:.0%}. Check the logs for the model's errors."
        )
    return results
