from __future__ import annotations

import pytest
from click.testing import CliRunner

from curator_agent.cli import main
from curator_agent.config import is_unset, required_env_for, resolve_provider


def test_rejects_non_board_urls():
    result = CliRunner().invoke(main, ["https://www.pinterest.com/pin/123/"])
    assert result.exit_code == 2
    assert "Invalid Pinterest board URL" in result.output


def test_rejects_bad_cluster_counts():
    result = CliRunner().invoke(main, ["https://www.pinterest.com/u/b/", "--clusters", "0"])
    assert result.exit_code == 2


def test_missing_credentials_exit_2(monkeypatch):
    result = CliRunner().invoke(main, ["https://www.pinterest.com/u/b/", "--provider", "gemini"])
    assert result.exit_code == 2
    assert "GEMINI_API_KEY" in result.output


@pytest.mark.parametrize("value, unset", [
    (None, True), ("", True), ("  ", True), ("your_gemini_api_key_here", True), ("AIza123", False),
])
def test_is_unset_treats_example_placeholders_as_missing(value, unset):
    assert is_unset(value) is unset


def test_resolve_provider_prefers_gemini_then_groq_then_local(monkeypatch):
    assert resolve_provider("auto") == "local"
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert resolve_provider("auto") == "groq"
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    assert resolve_provider("auto") == "groq"
    monkeypatch.setenv("GEMINI_API_KEY", "real")
    assert resolve_provider("auto") == "gemini"
    assert resolve_provider("local") == "local"


def test_required_env_for():
    assert required_env_for("local", "rss", None) == []
    assert required_env_for("gemini", "api", "db") == ["GEMINI_API_KEY", "PINTEREST_ACCESS_TOKEN", "NOTION_API_KEY"]
