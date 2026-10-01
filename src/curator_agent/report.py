"""Run outputs: machine-readable results.json and a self-contained report.html.

The HTML is rendered *from* the results dict, so the two can never disagree and
a report can be regenerated from an old results.json.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from curator_agent.models.cluster import MarketingStrategy, VisualCluster
from curator_agent.models.pin import Pin, PinMetadata
from curator_agent.sources import BoardRef


def _pin_url(pin: Pin, source: str) -> str | None:
    if source in ("rss", "api"):
        return f"https://www.pinterest.com/pin/{pin.id}/"
    return pin.source_link


def _image_src(pin: Pin) -> str:
    if pin.image_url.startswith(("http://", "https://")):
        return pin.image_url
    return Path(pin.image_url).resolve().as_uri()


def build_results(
    *,
    board: BoardRef,
    source: str,
    provider: str,
    model: str | None,
    run_id: str,
    pins: list[Pin],
    metadata: list[PinMetadata],
    clusters: list[VisualCluster],
    strategies: list[MarketingStrategy],
    k: int,
    silhouette: float | None,
    k_scores: dict[int, float],
) -> dict:
    pins_by_id = {p.id: p for p in pins}
    meta_by_id = {m.pin_id: m for m in metadata}
    strategy_by_cluster = {s.cluster_id: s for s in strategies}

    def pin_entry(pin_id: str) -> dict:
        pin, meta = pins_by_id[pin_id], meta_by_id[pin_id]
        return {
            "id": pin.id,
            "title": pin.title,
            "image_url": _image_src(pin),
            "pin_url": _pin_url(pin, source),
            "mood": meta.mood,
            "style": meta.style_category,
            "aesthetic_tags": meta.aesthetic_tags,
            "palette": meta.color_palette,
        }

    def strategy_entry(cluster_id: str) -> dict | None:
        s = strategy_by_cluster.get(cluster_id)
        if s is None:
            return None
        return {
            "brand_guidelines": s.brand_guidelines,
            "palette_recommendations": s.palette_recommendations,
            "content_calendar": s.content_calendar,
            "search_keywords": s.search_keywords,
        }

    return {
        "board": {"url": board.url, "board_id": board.board_id, "source": source},
        "run": {
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "metrics": {
            "pins_fetched": len(pins),
            "pins_analyzed": len(metadata),
            "k": k,
            "silhouette": None if silhouette is None else round(silhouette, 4),
            "k_scores": {str(kk): round(v, 4) for kk, v in sorted(k_scores.items())},
        },
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "theme_label": c.theme_label,
                "size": len(c.member_pin_ids),
                "cohesion": c.cohesion,
                "dominant_palette": c.dominant_palette,
                "distinctive_tags": c.representative_tags,
                "strategy": strategy_entry(c.cluster_id),
                "pins": [pin_entry(pid) for pid in c.member_pin_ids],
            }
            for c in clusters
        ],
    }


def write_outputs(results: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    html_path = out_dir / "report.html"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(results), encoding="utf-8")
    return json_path, html_path


# --------------------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#faf8f5;--card:#fff;--ink:#1f1d1a;--muted:#6b665e;--line:#e7e2da;--chip:#f1ede6;--accent:#b4452f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151412;--card:#1e1c1a;--ink:#f3efe8;--muted:#a59f95;--line:#34312d;--chip:#2a2825;--accent:#e2836c}}
:root[data-theme=dark]{--bg:#151412;--card:#1e1c1a;--ink:#f3efe8;--muted:#a59f95;--line:#34312d;--chip:#2a2825;--accent:#e2836c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:inherit}
.wrap{max-width:1180px;margin:0 auto;padding:40px 16px 80px}
.brand{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(28px,5vw,44px)/1.1 ui-serif,Georgia,serif;margin:8px 0 12px}
.meta{color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 18px}
.meta b{color:var(--ink);font-weight:600}
nav.themes{display:flex;flex-wrap:wrap;gap:8px;margin:28px 0 8px}
nav.themes a{display:flex;align-items:center;gap:8px;padding:7px 14px;border-radius:999px;background:var(--chip);text-decoration:none;font-size:14px}
.dots{display:inline-flex}.dots i{width:12px;height:12px;border-radius:50%;margin-left:-4px;border:2px solid var(--chip)}
.panel{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin:24px 0 8px}
@media (max-width:760px){.panel{grid-template-columns:1fr}}
.box{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.box h3{margin:0 0 10px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.kbars{display:flex;align-items:flex-end;gap:10px;height:96px;padding-top:6px}
.kbar{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;font-size:12px;color:var(--muted)}
.kbar span{width:100%;max-width:44px;background:var(--line);border-radius:6px 6px 2px 2px;min-height:3px}
.kbar.best span{background:var(--accent)}.kbar.best{color:var(--ink);font-weight:600}
section.cluster{margin-top:56px;padding-top:28px;border-top:1px solid var(--line)}
.chead{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px}
h2{font:600 clamp(22px,3.4vw,30px)/1.2 ui-serif,Georgia,serif;margin:0}
.stat{color:var(--muted);font-size:14px}
.palette{display:flex;border-radius:14px;overflow:hidden;margin:16px 0 12px;height:56px}
.palette div{flex:1;display:flex;align-items:flex-end;padding:6px 8px;font:11px ui-monospace,monospace}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{background:var(--chip);border-radius:999px;padding:3px 11px;font-size:13px}
.strategy{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;margin:18px 0}
@media (max-width:760px){.strategy{grid-template-columns:1fr}}
.strategy ol{margin:0;padding-left:20px}.strategy li{margin:6px 0}
.note{color:var(--muted);font-size:14px;font-style:italic;margin:14px 0}
.grid{columns:5 200px;column-gap:14px;margin-top:18px}
.pin{break-inside:avoid;margin:0 0 14px;display:block;text-decoration:none}
.pin img{width:100%;display:block;border-radius:16px;background:var(--chip)}
.pin p{margin:6px 4px 0;font-size:13px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
footer{margin-top:72px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:20px}
"""


def _text_color(hex_code: str) -> str:
    r, g, b = (int(hex_code[i:i + 2], 16) for i in (1, 3, 5))
    return "#1f1d1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def _palette_html(palette: list[str]) -> str:
    cells = "".join(
        f'<div style="background:{escape(c)};color:{_text_color(c)}">{escape(c)}</div>' for c in palette
    )
    return f'<div class="palette" role="img" aria-label="Palette {escape(" ".join(palette))}">{cells}</div>'


def _chips(items: list[str]) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{escape(i)}</span>' for i in items) + "</div>"


def _k_chart(k_scores: dict[str, float], k: int) -> str:
    if not k_scores:
        return '<p class="note">k was set manually.</p>'
    # Scores usually sit close together, so scale bars to the observed range.
    lo, hi = min(k_scores.values()), max(k_scores.values())
    span = (hi - lo) or 1.0
    bars = "".join(
        f'<div class="kbar{" best" if int(kk) == k else ""}" title="silhouette {v:.3f}">'
        f'<small>{v:.2f}</small><span style="height:{12 + (v - lo) / span * 58:.0f}px"></span>k={escape(kk)}</div>'
        for kk, v in k_scores.items()
    )
    return f'<div class="kbars">{bars}</div><p class="note" style="margin:8px 0 0">Bar = cosine silhouette score; the highest wins.</p>'


def _pin_html(pin: dict) -> str:
    title = pin.get("title") or ""
    img = f'<img src="{escape(pin["image_url"])}" alt="{escape(title[:120])}" loading="lazy">'
    caption = f"<p>{escape(title)}</p>" if title else ""
    if pin.get("pin_url"):
        return f'<a class="pin" href="{escape(pin["pin_url"])}" target="_blank" rel="noopener">{img}{caption}</a>'
    return f'<div class="pin">{img}{caption}</div>'


def _strategy_html(strategy: dict | None, provider: str) -> str:
    if strategy is None:
        if provider == "local":
            return '<p class="note">Run with a GEMINI_API_KEY or GROQ_API_KEY to add brand guidelines, a content calendar and search keywords for this theme.</p>'
        return '<p class="note">Strategy generation failed for this theme — see the run log.</p>'
    calendar = "".join(
        f"<li><b>{escape(i.get('theme', ''))}</b> — {escape(i.get('description', ''))}</li>"
        for i in strategy["content_calendar"]
    )
    return (
        '<div class="strategy">'
        f'<div class="box"><h3>Brand direction</h3><p>{escape(strategy["brand_guidelines"])}</p>'
        f'<h3>Pinterest search keywords</h3>{_chips(strategy.get("search_keywords", []))}</div>'
        f'<div class="box"><h3>Content calendar</h3><ol>{calendar}</ol></div>'
        "</div>"
    )


def render_html(results: dict) -> str:
    board, run, metrics = results["board"], results["run"], results["metrics"]
    title = board["board_id"].split("_", 1)[-1].replace("-", " ").title()
    sil = metrics["silhouette"]
    model = f" · {escape(run['model'])}" if run.get("model") else ""

    nav = "".join(
        f'<a href="#{escape(c["cluster_id"])}"><span class="dots">'
        + "".join(f'<i style="background:{escape(col)}"></i>' for col in c["dominant_palette"][:3])
        + f'</span>{escape(c["theme_label"])}</a>'
        for c in results["clusters"]
    )

    sections = []
    for c in results["clusters"]:
        sections.append(
            f'<section class="cluster" id="{escape(c["cluster_id"])}">'
            f'<div class="chead"><h2>{escape(c["theme_label"])}</h2>'
            f'<span class="stat">{c["size"]} pins · cohesion {c["cohesion"]:.2f}</span></div>'
            f'{_palette_html(c["dominant_palette"])}'
            f'{_chips(c["distinctive_tags"])}'
            f'{_strategy_html(c["strategy"], run["provider"])}'
            f'<div class="grid">{"".join(_pin_html(p) for p in c["pins"])}</div>'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · MuseBoard</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<div class="brand">MuseBoard report</div>
<h1>{escape(title)}</h1>
<div class="meta">
<span><b>{metrics["pins_analyzed"]}</b> pins analysed</span>
<span><b>{len(results["clusters"])}</b> visual themes</span>
<span>silhouette <b>{"n/a" if sil is None else f"{sil:.3f}"}</b></span>
<span>{escape(run["provider"])}{model} via {escape(board["source"])}</span>
<a href="{escape(board["url"])}" target="_blank" rel="noopener">View board ↗</a>
</div>
<nav class="themes">{nav}</nav>
<div class="panel">
<div class="box"><h3>Choosing the number of themes</h3>{_k_chart(metrics["k_scores"], metrics["k"])}</div>
<div class="box"><h3>How this was made</h3><p style="margin:0">Each pin is embedded with CLIP from its pixels{" and its AI description" if run["provider"] != "local" else ""}, grouped with k-means (k picked by cosine silhouette), and described by the tags most over-represented in each group. Palettes are k-means over real pixels.</p></div>
</div>
{"".join(sections)}
<footer>Generated {escape(run["generated_at"])} · run {escape(run["run_id"])} · <a href="https://github.com/nimnay/MuseBoard">MuseBoard</a></footer>
</div></body></html>
"""
