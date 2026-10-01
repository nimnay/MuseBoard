"""``museboard`` command-line entry point.

Exit codes: 0 success · 2 invalid arguments or missing credentials ·
3 a pipeline stage failed · 4 Notion export failed (results still on disk).
"""
from __future__ import annotations

import os
import sys
import uuid
import webbrowser
from pathlib import Path

import click
from dotenv import load_dotenv

from curator_agent.config import get_env, is_unset, required_env_for, resolve_provider
from curator_agent.sources import BoardRef, PinSource, board_ref_for_file, parse_board_url


def _parse_clusters(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    try:
        n = int(value)
    except ValueError:
        raise click.BadParameter("must be 'auto' or a positive integer") from None
    if n < 1:
        raise click.BadParameter("must be 'auto' or a positive integer")
    return n


def _resolve_source(board: str, source: str) -> tuple[BoardRef, PinSource]:
    from curator_agent.sources.api import ApiSource
    from curator_agent.sources.local import LocalSource
    from curator_agent.sources.rss import RssSource

    if source == "local" or (source == "auto" and Path(board).is_file()):
        path = Path(board)
        if not path.is_file():
            raise click.BadParameter(f"{board!r} is not a file", param_hint="BOARD")
        return board_ref_for_file(path), LocalSource(path)

    try:
        ref = parse_board_url(board)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="BOARD") from None

    token = get_env("PINTEREST_ACCESS_TOKEN")
    if source == "api" or (source == "auto" and token):
        return ref, ApiSource(token or "")
    return ref, RssSource()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("board")
@click.option("--source", type=click.Choice(["auto", "rss", "api", "local"]), default="auto", show_default=True,
              help="Where pins come from. auto: a local JSON path → local; PINTEREST_ACCESS_TOKEN set → api; else rss.")
@click.option("--provider", type=click.Choice(["auto", "gemini", "groq", "local"]), default="auto",
              show_default=True, envvar="MUSEBOARD_PROVIDER",
              help="Vision/LLM backend. auto picks Gemini, then Groq, then key-free local CLIP tagging.")
@click.option("--clusters", default="auto", show_default=True,
              help="Number of themes, or 'auto' to choose by silhouette score.")
@click.option("--max-pins", default=50, show_default=True, type=click.IntRange(1, 250))
@click.option("--notion-db", envvar="NOTION_DATABASE_ID", default=None,
              help="Also export each theme to this Notion database.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Output directory. [default: output/<board>]")
@click.option("--no-cache", is_flag=True, help="Refetch pins and re-run vision instead of using cache/.")
@click.option("--open", "open_report", is_flag=True, help="Open the HTML report when done.")
@click.option("--log-level", type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
              default="warning", show_default=True, help="Console verbosity (the run log always has INFO).")
def main(
    board: str, source: str, provider: str, clusters: str, max_pins: int, notion_db: str | None,
    out_dir: Path | None, no_cache: bool, open_report: bool, log_level: str,
) -> None:
    """Turn a Pinterest board into visual themes, palettes and a content plan.

    BOARD is a public board URL (https://www.pinterest.com/<user>/<board>/)
    or a path to a local pins JSON file.
    """
    n_clusters = _parse_clusters(clusters)
    board_ref, pin_source = _resolve_source(board, source)
    if is_unset(notion_db):
        notion_db = None
    resolved_provider = resolve_provider(provider)

    missing = [v for v in required_env_for(resolved_provider, pin_source.name, notion_db) if not get_env(v)]
    if missing:
        click.echo(f"ERROR: missing environment variables: {', '.join(missing)} (see .env.example)", err=True)
        sys.exit(2)

    os.environ["LOG_LEVEL"] = log_level.upper()
    from curator_agent.pipeline import Pipeline, RunConfig

    run_id = uuid.uuid4().hex[:12]
    click.echo(f"MuseBoard · {board_ref.url} · source={pin_source.name} · provider={resolved_provider} · run {run_id}\n")
    cfg = RunConfig(
        board=board_ref, source=pin_source, provider=resolved_provider, n_clusters=n_clusters,
        max_pins=max_pins, notion_db_id=notion_db, out_dir=out_dir, no_cache=no_cache,
    )
    exit_code = Pipeline(run_id).run(cfg)
    if exit_code == 0 and open_report:
        report = (out_dir or Path("output") / board_ref.board_id) / "report.html"
        webbrowser.open(report.resolve().as_uri())
    sys.exit(exit_code)


def entrypoint() -> None:
    # Windows pipes default to cp1252, which can't encode the ✓/→ in our output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv()
    main()
