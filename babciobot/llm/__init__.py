"""LLM client layer for babciobot.

A thin, config-driven wrapper around the OpenAI-compatible API (Ollama Cloud by default).
Keeps every model id and the base URL in ``config.toml`` so the endpoint/model can be
swapped with a one-line edit — the correctness escape hatch if no Ollama Cloud model
passes the golden-set benchmark (T-08).
"""

from babciobot.llm.client import (
    AuthError,
    BadJsonResponse,
    JobNotConfigured,
    LLMClient,
    LlmClientError,
    ModelNotAvailable,
)
from babciobot.llm.logging import LlmCallsLog
from babciobot.llm.schemas import Correction

__all__ = [
    "LLMClient",
    "LlmCallsLog",
    "Correction",
    "LlmClientError",
    "ModelNotAvailable",
    "AuthError",
    "JobNotConfigured",
    "BadJsonResponse",
]