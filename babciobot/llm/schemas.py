"""Pydantic schemas for LLM jobs.

Ollama Cloud has no structured-output support (T-04), so the client asks for JSON in the
prompt and validates the response against these models client-side. New jobs add a model
here and a ``[models]`` line in ``config.toml``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Correction(BaseModel):
    """Result of correcting a single Polish sentence.

    Attributes:
        corrected: The corrected Polish sentence.
        explanation: A short note on what was wrong (Polish or English).
    """

    corrected: str = Field(description="The corrected Polish sentence.")
    explanation: str = Field(description="Short note on what was wrong and why.")