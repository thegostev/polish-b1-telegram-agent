"""Tests for babciobot.llm.logging (the llm_calls SQLite log)."""

from __future__ import annotations

from pathlib import Path

import pytest

from babciobot.llm.logging import PREVIEW_LEN, LlmCallsLog


def test_creates_table_idempotent(tmp_path):
    db = tmp_path / "calls.db"
    # Opening twice on the same path must not error (CREATE TABLE IF NOT EXISTS).
    with LlmCallsLog(db):
        pass
    with LlmCallsLog(db):
        pass
    assert db.exists()


def test_log_call_and_recent(tmp_path):
    db = tmp_path / "calls.db"
    log = LlmCallsLog(db)
    log.log_call(
        job="correction", model="m", attempt=1, status="ok",
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        latency_ms=42, error=None, response_preview="hello",
    )
    log.log_call(
        job="correction", model="m", attempt=1, status="error",
        prompt_tokens=None, completion_tokens=None, total_tokens=None,
        latency_ms=500, error="boom", response_preview=None,
    )
    rows = log.recent(limit=10)
    count = log.count()
    log.close()

    assert count == 2
    assert len(rows) == 2
    # newest first
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "boom"
    assert rows[1]["status"] == "ok"
    assert rows[1]["total_tokens"] == 15
    assert rows[1]["latency_ms"] == 42
    assert rows[1]["response_preview"] == "hello"


def test_preview_truncated(tmp_path):
    db = tmp_path / "calls.db"
    long_text = "x" * (PREVIEW_LEN + 50)
    with LlmCallsLog(db) as log:
        log.log_call(
            job="correction", model="m", attempt=1, status="ok",
            prompt_tokens=None, completion_tokens=None, total_tokens=None,
            latency_ms=1, error=None, response_preview=long_text,
        )
        rows = log.recent(limit=1)
    assert len(rows[0]["response_preview"]) == PREVIEW_LEN


def test_parent_dir_created(tmp_path):
    db = tmp_path / "nested" / "deeper" / "calls.db"
    assert not db.parent.exists()
    log = LlmCallsLog(db)
    assert db.parent.exists()
    log.log_call(
        job="correction", model="m", attempt=1, status="ok",
        prompt_tokens=None, completion_tokens=None, total_tokens=None,
        latency_ms=1, error=None, response_preview=None,
    )
    assert log.count() == 1
    log.close()