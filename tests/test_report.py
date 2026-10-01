from __future__ import annotations

from curator_agent.report import render_html


def _results(**overrides) -> dict:
    results = {
        "board": {"url": "https://www.pinterest.com/u/cozy-kitchens/", "board_id": "u_cozy-kitchens", "source": "rss"},
        "run": {"run_id": "r1", "provider": "local", "model": None, "generated_at": "2026-10-01T00:00:00+00:00"},
        "metrics": {"pins_fetched": 2, "pins_analyzed": 2, "k": 2, "silhouette": 0.31,
                    "k_scores": {"2": 0.31, "3": 0.2}},
        "clusters": [{
            "cluster_id": "cluster_0", "theme_label": "Warm Oak", "size": 1, "cohesion": 0.9,
            "dominant_palette": ["#FFFFFF", "#000000"], "distinctive_tags": ["oak"], "strategy": None,
            "pins": [{"id": "1", "title": "<script>alert(1)</script>", "image_url": "https://i/1.jpg",
                      "pin_url": "https://www.pinterest.com/pin/1/"}],
        }],
    }
    results.update(overrides)
    return results


def test_report_escapes_untrusted_pin_text():
    html = render_html(_results())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_contains_board_title_themes_and_k_chart():
    html = render_html(_results())
    assert "<title>Cozy Kitchens · MuseBoard</title>" in html
    assert 'href="#cluster_0"' in html
    assert 'class="kbar best"' in html
    assert "GEMINI_API_KEY" in html  # local runs explain how to get strategies


def test_report_renders_strategy():
    results = _results()
    results["clusters"][0]["strategy"] = {
        "brand_guidelines": "Slow & warm", "palette_recommendations": [],
        "content_calendar": [{"theme": "W1", "description": "Shelf styling"}],
        "search_keywords": ["oak shelves"],
    }
    html = render_html(results)
    assert "Slow &amp; warm" in html and "oak shelves" in html and "Shelf styling" in html
