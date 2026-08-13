"""Config loading for babciobot.

Everything changeable lives in ``config.toml`` (TOML, read with stdlib ``tomllib``) and the
``.env`` secrets file — nothing model- or endpoint-related is hardcoded. ``load_config()``
returns a typed ``Config``; ``get_api_key()`` resolves ``OLLAMA_API_KEY`` from the
environment, loading the ``.env`` file first if needed.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """Raised when config is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class LlmConfig:
    """LLM transport settings — the one retry/timeout policy."""

    base_url: str
    timeout_seconds: float
    max_retries: int  # retry count for transport + JSON-validation failures
    retry_backoff_seconds: float


@dataclass(frozen=True)
class SecretsConfig:
    """Path to the ``.env`` file holding ``OLLAMA_API_KEY`` (outside the repo/vault)."""

    env_path: Path


@dataclass(frozen=True)
class StorageConfig:
    """Where the SQLite ``llm_calls`` log lives."""

    db_path: Path  # absolute, resolved against the repo root


@dataclass(frozen=True)
class Config:
    """Resolved config — all knobs from ``config.toml`` plus the repo root."""

    repo_root: Path
    llm: LlmConfig
    secrets: SecretsConfig
    storage: StorageConfig
    models: dict[str, str] = field(default_factory=dict)

    def model_for(self, job: str) -> str:
        """Return the configured model id for a job, or raise ConfigError."""
        try:
            return self.models[job]
        except KeyError as exc:
            known = ", ".join(sorted(self.models)) or "(none configured)"
            raise ConfigError(
                f"No model configured for job {job!r}. "
                f"Add a [models] line in config.toml. Known jobs: {known}."
            ) from exc


def _load_dotenv(path: Path, *, overwrite: bool = False) -> None:
    """Parse a minimal KEY=VALUE .env file into os.environ.

    Skips blank lines and ``#`` comments. Existing env vars are preserved unless
    ``overwrite`` is set. Values may be bare or single/double-quoted.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if overwrite or key not in os.environ:
            os.environ[key] = value


def get_api_key(config: Config, *, env_var: str = "OLLAMA_API_KEY") -> str:
    """Resolve the API key from the environment, loading the .env file if needed.

    Raises:
        ConfigError: if the key is absent after loading the .env file.
    """
    if env_var not in os.environ:
        _load_dotenv(config.secrets.env_path)
    key = os.environ.get(env_var)
    if not key:
        raise ConfigError(
            f"{env_var} not set. Either export it or add it to "
            f"{config.secrets.env_path} (see .env.example)."
        )
    return key


def load_config(
    path: str | Path = "config.toml",
    *,
    load_secrets: bool = True,
) -> Config:
    """Load and validate config.toml into a Config.

    Args:
        path: Path to the config file. Relative paths resolve against CWD.
        load_secrets: If True (default), load the .env file so OLLAMA_API_KEY is
            available to ``get_api_key``. Tests may pass False to isolate env state.

    Raises:
        ConfigError: if the file is missing, unparseable, or missing required keys.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    repo_root = config_path.parent

    try:
        llm_raw = raw["llm"]
        llm = LlmConfig(
            base_url=str(llm_raw["base_url"]),
            timeout_seconds=float(llm_raw["timeout_seconds"]),
            max_retries=int(llm_raw["max_retries"]),
            retry_backoff_seconds=float(llm_raw["retry_backoff_seconds"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing [llm] key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Bad [llm] value: {exc}") from exc

    secrets_raw = raw.get("secrets", {})
    secrets = SecretsConfig(env_path=Path(str(secrets_raw.get("env_path", ""))).expanduser())

    storage_raw = raw.get("storage", {})
    db_path = Path(str(storage_raw.get("db_path", "data/babciobot.db")))
    if not db_path.is_absolute():
        db_path = repo_root / db_path
    storage = StorageConfig(db_path=db_path)

    models = {str(k): str(v) for k, v in raw.get("models", {}).items()}

    if load_secrets:
        _load_dotenv(secrets.env_path)

    return Config(
        repo_root=repo_root,
        llm=llm,
        secrets=secrets,
        storage=storage,
        models=models,
    )