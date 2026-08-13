# babciobot

Telegram bot coaching Alex and Olga toward a Polish B1 certificate (TELC B1-B2 Dual).
Built ticket by ticket; this snapshot implements **FYR-187** — the LLM client layer only.

## What's here

A config-driven LLM client against the OpenAI-compatible API (Ollama Cloud by default):

- `babciobot/config.py` — loads `config.toml` + resolves `OLLAMA_API_KEY` from env / `.env`.
- `babciobot/llm/client.py` — `LLMClient` with `complete()` / `complete_json()`, one
  retry/timeout policy, per-attempt logging, graceful model-not-found.
- `babciobot/llm/logging.py` — the `llm_calls` SQLite log (tokens, latency, model, status).
- `babciobot/llm/schemas.py` — Pydantic schemas for each job (`Correction` for the smoke test).
- `scripts/correct.py` — CLI smoke test: Polish sentence → correction via config model, logged.

Every model id and the base URL live in `config.toml` — swap a model or endpoint with a
one-line edit, no code change. That is the correctness escape hatch: if no Ollama Cloud
model passes the golden-set benchmark (T-08), grading points elsewhere here.

## Setup

```bash
# Homebrew python@3.13 is currently broken; use 3.12.
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# secrets stay outside the repo/vault (chmod 600). See .env.example.
cp .env.example ~/Projects/babciobot/.env   # then edit in the real OLLAMA_API_KEY
chmod 600 ~/Projects/babciobot/.env
```

## Run the smoke test (the FYR-187 "done when")

```bash
.venv/bin/python scripts/correct.py "Ja jest w domu"
# eyeball the corrected sentence + explanation, then check the log:
sqlite3 data/babciobot.db "select job,model,attempt,status,total_tokens,latency_ms from llm_calls order by id desc limit 3"
```

## Test

```bash
.venv/bin/python -m pytest tests/unit -v          # offline, no API key
.venm/bin/python -m pytest tests -m live -v        # live: needs OLLAMA_API_KEY + network
```

## Config knobs (changeable in ~30s, all in `config.toml`)

| Knob | Section | What it does |
| --- | --- | --- |
| `base_url` | `[llm]` | The OpenAI-compatible endpoint (escape hatch). |
| `timeout_seconds` | `[llm]` | Per-call timeout. |
| `max_retries` | `[llm]` | Retries for transport + JSON-validation failures. |
| `retry_backoff_seconds` | `[llm]` | Sleep between retries. |
| `env_path` | `[secrets]` | Where `OLLAMA_API_KEY` is loaded from. |
| `db_path` | `[storage]` | The `llm_calls` SQLite log location. |
| `correction` / `grading` | `[models]` | Per-job model choice. |

## Repo + secrets layout

- **Code**: this folder (vault, GitHub-linked).
- **Secrets**: `~/Projects/babciobot/.env` (outside iCloud, chmod 600, gitignored).

> The Telegram bot skeleton, full core SQLite schema, and launchd service come later
> (T-03 / T-06). The 5-line "how to update and restart" procedure will land with T-06.