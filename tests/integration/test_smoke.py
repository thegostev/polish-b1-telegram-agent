"""Live integration test — hits the real Ollama Cloud API.

Skipped unless OLLAMA_API_KEY is present and the ``live`` marker is selected
(``pytest -m live``). Costs one short call against the configured model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from babciobot.config import load_config
from babciobot.llm import Correction, LLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONFIG = REPO_ROOT / "config.toml"

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _skip_without_key():
    if not os.environ.get("OLLAMA_API_KEY"):
        pytest.skip("OLLAMA_API_KEY not set; set it or load ~/Projects/babciobot/.env")


def test_correction_round_trip_logs_a_row():
    """The FYR-187 'done when': a Polish sentence → correction via the config model, logged."""
    config = load_config(REPO_CONFIG)
    before = None
    client = LLMClient(config)
    try:
        # Count existing rows so we can assert a new one was added.
        before = client._log.count()
        result = client.complete_json(
            "correction",
            [
                {
                    "role": "system",
                    "content": "Jesteś korektorem języka polskiego. Popraw zdanie.",
                },
                {"role": "user", "content": "Ja jest w domu"},
            ],
            Correction,
        )
    finally:
        after = client._log.count()
        client.close()

    assert isinstance(result, Correction)
    assert result.corrected  # non-empty correction
    assert after == (before or 0) + 1  # exactly one new attempt row