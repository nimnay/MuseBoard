from __future__ import annotations

from PIL import Image

from curator_agent.images import extract_palette, load_image_bytes, merge_palettes, open_rgb


def test_extract_palette_orders_colors_by_area():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.paste((0, 0, 255), (0, 0, 100, 25))  # 25% blue, 75% red
    assert extract_palette(img, k=2) == ["#FF0000", "#0000FF"]


def test_extract_palette_handles_single_color_images():
    assert extract_palette(Image.new("RGB", (10, 10), (18, 52, 86)), k=5) == ["#123456"]


def test_merge_palettes_finds_shared_dominant_colors():
    merged = merge_palettes([["#FF0000", "#0000FF"], ["#FE0101", "#0000FF"], ["#FF0000"]], k=2)
    # Red leads three palettes, so it outranks blue.
    assert merged == ["#FF0000", "#0000FF"]
    assert merge_palettes([]) == []


def test_load_image_bytes_reads_local_files(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (4, 4), (1, 2, 3, 0)).save(path)
    img = open_rgb(load_image_bytes(str(path)))
    assert img.mode == "RGB" and img.size == (4, 4)
