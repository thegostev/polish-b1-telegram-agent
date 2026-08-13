"""LLM client — config-driven, one retry/timeout policy, every attempt logged.

Wraps the OpenAI-compatible API (Ollama Cloud by default). Design points, all from T-04:

* Base URL, model id, and per-job model choice live in ``config.toml`` — never in code.
  Swapping the endpoint or model is a one-line config edit (the correctness escape hatch).
* One retry/timeout policy, owned here: the OpenAI SDK's own retry is disabled
  (``max_retries=0``) so there is a single loop covering both transport failures and
  JSON-validation failures.
* Ollama Cloud has no structured-output support, so ``complete_json`` asks for JSON in the
  prompt and validates with a Pydantic model client-side, retrying once on a bad response.
* Cloud models get retired, so ``NotFoundError`` is mapped to a clear ``ModelNotAvailable``.
* Every attempt (success or failure) is written to the ``llm_calls`` log with tokens,
  latency, and status — cost and slowness stay visible later.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from babciobot.config import Config, get_api_key
from babciobot.llm.logging import LlmCallsLog

T = TypeVar("T", bound=BaseModel)

# Transport errors worth retrying (transient).
_RETRYABLE_TRANSPORT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LlmClientError(Exception):
    """Base class for LLM client failures."""


class ModelNotAvailable(LlmClientError):
    """The configured model was not found (likely retired or misspelled)."""


class AuthError(LlmClientError):
    """Authentication failed — bad or missing API key."""


class JobNotConfigured(LlmClientError):
    """No model is configured for the requested job."""


class BadJsonResponse(LlmClientError):
    """The model response could not be parsed as JSON for the schema after retries."""


def _strip_fences(text: str) -> str:
    """Strip a single pair of leading/trailing markdown code fences if present."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        else:
            s = s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _usage_tokens(usage: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Pull token counts from an OpenAI usage object, tolerating ``None``."""
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )


def _json_instruction(schema_model: type[BaseModel]) -> str:
    """Build the prompt instruction asking for schema-conformant JSON only."""
    schema = schema_model.model_json_schema()
    return (
        "Respond with ONLY valid JSON matching this JSON schema. "
        "No prose, no markdown code fences, no commentary before or after.\n\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


class LLMClient:
    """Config-driven OpenAI-compatible LLM client with retry + per-attempt logging.

    Args:
        config: Loaded ``Config``.
        api_key: Override the API key (else resolved from env/.env via ``get_api_key``).
        client: Inject an OpenAI-compatible client (tests pass a fake). If None, a real
            ``openai.OpenAI`` is built with ``max_retries=0`` so this class owns retries.
        log: Inject an ``LlmCallsLog`` (tests use a temp DB). If None, one is created
            from ``config.storage.db_path``.
    """

    def __init__(
        self,
        config: Config,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        log: Optional[LlmCallsLog] = None,
    ) -> None:
        self.config = config
        self.max_retries = config.llm.max_retries
        self.backoff_seconds = config.llm.retry_backoff_seconds
        if client is not None:
            self._client = client
        else:
            if api_key is None:
                api_key = get_api_key(config)
            self._client = openai.OpenAI(
                base_url=config.llm.base_url,
                api_key=api_key,
                max_retries=0,
                timeout=config.llm.timeout_seconds,
            )
        self._log = log if log is not None else LlmCallsLog(config.storage.db_path)

    def complete(
        self,
        job: str,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
    ) -> str:
        """Run a completion for ``job`` and return the raw text content."""
        return self._complete(job, messages, model=model, parse=lambda c: c)

    def complete_json(
        self,
        job: str,
        messages: list[dict[str, str]],
        schema_model: type[T],
        *,
        model: Optional[str] = None,
    ) -> T:
        """Run a completion for ``job`` and return a validated ``schema_model`` instance.

        Appends a prompt instruction to emit JSON matching ``schema_model``'s schema, then
        parses and validates the response. Retries once (per ``max_retries``) on a
        transport failure or a parse/validation failure.
        """
        instruction = _json_instruction(schema_model)
        full_messages = list(messages) + [{"role": "system", "content": instruction}]

        def parse(content: str) -> T:
            data = json.loads(_strip_fences(content))
            return schema_model.model_validate(data)

        return self._complete(job, full_messages, model=model, parse=parse)

    def _resolve_model(self, job: str, model: Optional[str]) -> str:
        if model is not None:
            return model
        try:
            return self.config.model_for(job)
        except Exception as exc:  # ConfigError
            raise JobNotConfigured(str(exc)) from exc

    def _complete(
        self,
        job: str,
        messages: list[dict[str, str]],
        *,
        model: Optional[str],
        parse: Any,
    ) -> Any:
        model_id = self._resolve_model(job, model)
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            start = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=model_id, messages=messages
                )
            except openai.NotFoundError as exc:
                self._log_attempt(
                    job, model_id, attempt, start, status="error", error=str(exc)
                )
                raise ModelNotAvailable(
                    f"Model {model_id!r} not found — it may be retired or misspelled. "
                    "Update [models] in config.toml."
                ) from exc
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                self._log_attempt(
                    job, model_id, attempt, start, status="error", error=str(exc)
                )
                raise AuthError(f"Authentication failed: {exc}") from exc
            except _RETRYABLE_TRANSPORT as exc:
                self._log_attempt(
                    job, model_id, attempt, start, status="error", error=str(exc)
                )
                if attempt <= self.max_retries:
                    self._sleep()
                    continue
                raise LlmClientError(
                    f"{model_id}: transport failed after {attempt} attempt(s): {exc}"
                ) from exc
            except openai.APIError as exc:
                # Non-retryable API error (e.g. BadRequestError) — don't loop.
                self._log_attempt(
                    job, model_id, attempt, start, status="error", error=str(exc)
                )
                raise LlmClientError(f"{model_id}: {exc}") from exc

            # Transport succeeded — extract content + tokens.
            latency_ms = int((time.monotonic() - start) * 1000)
            content = resp.choices[0].message.content
            p_tok, c_tok, t_tok = _usage_tokens(resp.usage)

            try:
                result = parse(content)
            except (json.JSONDecodeError, ValidationError) as exc:
                self._log_attempt(
                    job,
                    model_id,
                    attempt,
                    start,
                    status="error",
                    error=f"bad json: {exc}",
                    response_preview=content,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                )
                if attempt <= self.max_retries:
                    self._sleep()
                    continue
                raise BadJsonResponse(
                    f"{model_id}: response was not valid JSON for the schema "
                    f"after {attempt} attempt(s): {exc}"
                ) from exc

            self._log_attempt(
                job,
                model_id,
                attempt,
                start,
                status="ok",
                response_preview=content,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
            )
            return result

        # Unreachable: the loop always returns or raises.
        raise LlmClientError("unreachable")  # pragma: no cover

    def _log_attempt(
        self,
        job: str,
        model: str,
        attempt: int,
        start: float,
        *,
        status: str,
        error: Optional[str] = None,
        response_preview: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        latency_ms = int((time.monotonic() - start) * 1000)
        self._log.log_call(
            job=job,
            model=model,
            attempt=attempt,
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            error=error,
            response_preview=response_preview,
        )

    def _sleep(self) -> None:
        if self.backoff_seconds > 0:
            time.sleep(self.backoff_seconds)

    def close(self) -> None:
        self._log.close()