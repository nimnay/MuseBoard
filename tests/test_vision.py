from __future__ import annotations

from dataclasses import asdict

import jsonschema
import numpy as np
import pytest
from PIL import Image

from curator_agent.models.pin import Pin
from curator_agent.stages.vision import (
    SCHEMA,
    LocalTagger,
    extract_vision_metadata,
    validate_response,
)
from tests.conftest import VISION_RESPONSE, FakeLLM

SCHEMA_FIELDS = set(SCHEMA["properties"])


def _pins(n: int) -> tuple[list[Pin], dict]:
    pins = [Pin(id=f"p{i}", image_url=f"https://x/{i}.jpg", board_id="b") for i in range(n)]
    images = {p.id: Image.new("RGB", (16, 16), (200, 100, 50)) for p in pins}
    return pins, images


def test_validate_response_normalizes_case_and_uses_pixel_palette():
    meta = validate_response(dict(VISION_RESPONSE), "p1", ["#123456"])
    assert meta.aesthetic_tags == ["cozy", "minimalist"]
    assert meta.mood == "serene"
    assert meta.color_palette == ["#123456"]  # pixels win over the model's guess


def test_validate_response_rejects_schema_violations():
    bad = dict(VISION_RESPONSE, mood="")
    with pytest.raises(jsonschema.ValidationError):
        validate_response(bad, "p1", [])
    with pytest.raises(jsonschema.ValidationError):
        validate_response(dict(VISION_RESPONSE, surprise="field"), "p1", [])


def test_local_tagger_output_satisfies_the_vision_schema(encoder):
    img = Image.new("RGB", (16, 16), (10, 120, 40))
    vec = encoder.encode_images([img])[0]
    meta = LocalTagger(encoder).tag("p1", vec, ["#0A7828"])
    payload = {k: v for k, v in asdict(meta).items() if k in SCHEMA_FIELDS}
    jsonschema.validate(payload, SCHEMA)
    assert len(meta.aesthetic_tags) == 3


def test_llm_vision_runs_and_caches(encoder):
    pins, images = _pins(3)
    vecs = dict(zip(images, encoder.encode_images(list(images.values()))))
    llm = FakeLLM()
    first = extract_vision_metadata(pins, images, vecs, llm, encoder, "run")
    assert len(first) == 3 and llm.calls == 3
    second = extract_vision_metadata(pins, images, vecs, llm, encoder, "run")
    assert [m.pin_id for m in second] == ["p0", "p1", "p2"]
    assert llm.calls == 3  # served from cache/vision


def test_validity_gate_halts_when_too_many_pins_fail(encoder):
    pins, images = _pins(4)
    vecs = {k: np.zeros(512) for k in images}
    llm = FakeLLM(vision_response={"mood": "only a mood"})
    with pytest.raises(RuntimeError, match="0 of 4"):
        extract_vision_metadata(pins, images, vecs, llm, encoder, "run")
