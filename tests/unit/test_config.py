"""Tests for babciobot.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from babciobot.config import ConfigError, _load_dotenv, get_api_key, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONFIG = REPO_ROOT / "config.toml"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_real_config_toml():
    """The shipped config.toml must be valid and carry the two configured jobs."""
    cfg = load_config(REPO_CONFIG, load_secrets=False)
    assert cfg.llm.base_url == "https://ollama.com/v1"
    assert cfg.llm.max_retries >= 1
    # Both jobs configured with non-empty model ids (exact ids verified live in step 0,
    # not asserted here to keep the unit test offline and non-brittle).
    assert cfg.models.get("correction")
    assert cfg.models.get("grading")
    # db_path resolves to an absolute path under the repo root.
    assert cfg.storage.db_path.is_absolute()
    assert cfg.storage.db_path.name == "babciobot.db"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", load_secrets=False)


def test_bad_toml_raises(tmp_path):
    path = _write(tmp_path / "bad.toml", "this is = = not toml")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path, load_secrets=False)


def test_missing_llm_key_raises(tmp_path):
    path = _write(tmp_path / "c.toml", "[storage]\ndb_path = \"x.db\"\n")
    with pytest.raises(ConfigError, match="llm"):
        load_config(path, load_secrets=False)


def test_model_for_returns_value(tmp_config):
    assert tmp_config.model_for("correction") == "fake-model"
    assert tmp_config.model_for("grading") == "fake-grader"


def test_model_for_unknown_job_raises(tmp_config):
    with pytest.raises(ConfigError, match="No model configured"):
        tmp_config.model_for("essay-grading")


def test_relative_db_path_resolves_against_repo_root(tmp_path):
    path = _write(
        tmp_path / "c.toml",
        "[llm]\nbase_url='x'\ntimeout_seconds=1\nmax_retries=0\nretry_backoff_seconds=0.0\n"
        "[storage]\ndb_path='logs/calls.db'\n",
    )
    cfg = load_config(path, load_secrets=False)
    assert cfg.storage.db_path == path.parent / "logs" / "calls.db"


def test_load_dotenv_sets_missing_var_and_respects_existing(tmp_path, monkeypatch):
    env_file = _write(tmp_path / ".env", "# a comment\nOLLAMA_API_KEY='sk-test-123'\nOTHER=val\n")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    _load_dotenv(env_file)
    assert os.environ["OLLAMA_API_KEY"] == "sk-test-123"
    assert os.environ["OTHER"] == "val"

    # Existing env vars are not overwritten.
    monkeypatch.setenv("OTHER", "already-set")
    _load_dotenv(env_file)
    assert os.environ["OTHER"] == "already-set"


def test_get_api_key_from_env(tmp_config, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-from-env")
    assert get_api_key(tmp_config) == "sk-from-env"


def test_get_api_key_missing_raises(tmp_config, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    # The fixture's env_path (tmp_path / ".env") does not exist, so nothing loads.
    with pytest.raises(ConfigError, match="not set"):
        get_api_key(tmp_config)