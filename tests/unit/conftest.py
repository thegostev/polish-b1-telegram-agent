"""Shared fixtures for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Put the repo root on sys.path so `import babciobot` works without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babciobot.config import Config, LlmConfig, SecretsConfig, StorageConfig


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A Config pointed at temp paths, with two configured jobs."""
    return Config(
        repo_root=tmp_path,
        llm=LlmConfig(
            base_url="https://example.invalid/v1",
            timeout_seconds=30,
            max_retries=1,
            retry_backoff_seconds=0.0,  # no real sleeping in tests
        ),
        secrets=SecretsConfig(env_path=tmp_path / ".env"),
        storage=StorageConfig(db_path=tmp_path / "data" / "test.db"),
        models={"correction": "fake-model", "grading": "fake-grader"},
    )