"""Tests for babciobot.llm.client — retry policy, JSON validation, model-not-found, logging.

Uses a fake OpenAI-compatible client injected via ``LLMClient(client=...)`` so no network
and no real API key are needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import openai
import pytest

from babciobot.llm import (
    AuthError,
    BadJsonResponse,
    Correction,
    JobNotConfigured,
    LLMClient,
    ModelNotAvailable,
)
from babciobot.llm.logging import LlmCallsLog


class _Usage(SimpleNamespace):
    def __init__(self, p: int, c: int, t: int) -> None:
        super().__init__(prompt_tokens=p, completion_tokens=c, total_tokens=t)


class _Response(SimpleNamespace):
    def __init__(self, content: str, usage: Any | None = None) -> None:
        super().__init__(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )


class _Completions:
    """Records calls and yields the next behavior (content string, (content, usage), or Exception)."""

    def __init__(self, behaviors: list[Any]) -> None:
        self._iter = iter(behaviors)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def create(self, model: str, messages: list[dict[str, str]]) -> Any:
        self.calls.append((model, messages))
        behavior = next(self._iter)
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, tuple):
            content, usage = behavior
        else:
            content, usage = behavior, _Usage(1, 2, 3)
        return _Response(content, usage)


def _fake_client(behaviors: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(behaviors)))


def _client(cfg, behaviors, log):
    return LLMClient(cfg, client=_fake_client(behaviors), log=log), log


# --- Exception fakes: subclasses that bypass the openai SDK's __init__ (which needs an httpx.Response). ---


class _FakeNotFound(openai.NotFoundError):
    def __init__(self, message: str = "model not found") -> None:
        Exception.__init__(self, message)


class _FakeAuth(openai.AuthenticationError):
    def __init__(self, message: str = "bad key") -> None:
        Exception.__init__(self, message)


class _FakeTimeout(openai.APITimeoutError):
    def __init__(self, message: str = "timeout") -> None:
        Exception.__init__(self, message)


# --- happy paths ---


def test_complete_returns_content(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [("poprawione", _Usage(7, 3, 10))], log)
    result = client.complete("correction", [{"role": "user", "content": "hi"}])
    assert result == "poprawione"
    rows = log.recent(1)
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempt"] == 1
    assert rows[0]["total_tokens"] == 10
    client.close()


def test_complete_json_happy_path(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    payload = json.dumps({"corrected": "Jestem w domu", "explanation": "jest→jestem"})
    client, _ = _client(tmp_config, [payload], log)
    result = client.complete_json(
        "correction",
        [{"role": "user", "content": "Ja jest w domu"}],
        Correction,
    )
    assert isinstance(result, Correction)
    assert result.corrected == "Jestem w domu"
    assert log.recent(1)[0]["status"] == "ok"
    client.close()


def test_complete_json_strips_code_fences(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    fenced = "```json\n" + json.dumps({"corrected": "x", "explanation": "y"}) + "\n```"
    client, _ = _client(tmp_config, [fenced], log)
    result = client.complete_json("correction", [{"role": "user", "content": "x"}], Correction)
    assert result.corrected == "x"
    client.close()


# --- retries ---


def test_complete_json_retries_on_bad_then_good(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    good = json.dumps({"corrected": "ok", "explanation": "fixed"})
    client, _ = _client(tmp_config, ["not json at all", good], log)
    result = client.complete_json("correction", [{"role": "user", "content": "x"}], Correction)
    assert result.corrected == "ok"
    rows = log.recent(10)
    # two attempts: first error, second ok
    assert len(rows) == 2
    assert rows[1]["status"] == "error"  # newer first → second attempt is rows[0]
    assert rows[1]["attempt"] == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempt"] == 2
    client.close()


def test_complete_json_exhausts_retries_and_raises(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, ["nope", "still nope"], log)
    with pytest.raises(BadJsonResponse):
        client.complete_json("correction", [{"role": "user", "content": "x"}], Correction)
    assert log.count() == 2
    assert all(r["status"] == "error" for r in log.recent(2))
    client.close()


def test_transport_retry_then_success(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [_FakeTimeout(), "recovered"], log)
    result = client.complete("correction", [{"role": "user", "content": "x"}])
    assert result == "recovered"
    rows = log.recent(2)
    assert rows[1]["status"] == "error"
    assert rows[0]["status"] == "ok"
    client.close()


def test_transport_exhausts_retries_and_raises(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [_FakeTimeout(), _FakeTimeout()], log)
    from babciobot.llm.client import LlmClientError

    with pytest.raises(LlmClientError):
        client.complete("correction", [{"role": "user", "content": "x"}])
    assert log.count() == 2
    client.close()


# --- non-retryable errors ---


def test_model_not_found_raises_model_not_available(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [_FakeNotFound()], log)
    with pytest.raises(ModelNotAvailable, match="retired"):
        client.complete("correction", [{"role": "user", "content": "x"}])
    assert log.count() == 1  # logged once, not retried
    client.close()


def test_auth_error_raises_auth_error(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [_FakeAuth()], log)
    with pytest.raises(AuthError):
        client.complete("correction", [{"role": "user", "content": "x"}])
    assert log.count() == 1
    client.close()


# --- config resolution ---


def test_job_not_configured_raises_before_any_call(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client = LLMClient(tmp_config, client=_fake_client([]), log=log)
    with pytest.raises(JobNotConfigured):
        client.complete("essay-grading", [{"role": "user", "content": "x"}])
    # No API call should have been made.
    assert client._client.chat.completions.calls == []
    client.close()


def test_model_override_used(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    client, _ = _client(tmp_config, [("ok", _Usage(1, 1, 2))], log)
    client.complete("correction", [{"role": "user", "content": "x"}], model="override-model")
    assert client._client.chat.completions.calls[0][0] == "override-model"
    client.close()


def test_json_instruction_appended_to_messages(tmp_config):
    log = LlmCallsLog(tmp_config.storage.db_path)
    payload = json.dumps({"corrected": "x", "explanation": "y"})
    client, _ = _client(tmp_config, [payload], log)
    client.complete_json("correction", [{"role": "user", "content": "x"}], Correction)
    sent = client._client.chat.completions.calls[0][1]
    # Original user message plus a trailing system instruction.
    assert sent[0]["role"] == "user"
    assert sent[-1]["role"] == "system"
    assert "valid JSON" in sent[-1]["content"]
    client.close()