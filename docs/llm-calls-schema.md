# `llm_calls` SQLite schema

The `llm_calls` table is created by `babciobot/llm/logging.py` with
`CREATE TABLE IF NOT EXISTS`. It is intentionally narrow and pedagogy-agnostic.

T-06 (bot skeleton) will fold this table into the bot's core schema
(`learners`, `sessions`, `messages`, `llm_calls`, `config`). This document is the
canonical DDL so the two definitions match.

## DDL

```sql
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,          -- ISO8601 UTC
    job TEXT NOT NULL,                 -- "correction", "grading", ...
    model TEXT NOT NULL,
    attempt INTEGER NOT NULL,          -- 1-based; >1 means a retry fired
    status TEXT NOT NULL,              -- "ok" | "error"
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    error TEXT,                        -- NULL on ok
    response_preview TEXT              -- first ~200 chars, for eyeballing
);
```

## Why every attempt is logged

Cost and slowness are the two things you can't see after the fact unless you recorded them.
A failed first attempt that succeeded on retry still cost tokens and latency, so each
attempt is a row — `attempt > 1` surfaces retries, `latency_ms` surfaces slow models, and the
token columns surface cost against the Ollama Cloud subscription caps (5-hour session /
7-day weekly limits per T-04).