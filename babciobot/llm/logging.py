"""SQLite ``llm_calls`` log — makes cost and slowness visible after the fact.

Every call attempt (success and failure) is recorded with model, token counts, latency,
and status, so retries and slow models show up later. The schema is intentionally narrow
and pedagogy-agnostic; T-06 will fold this table into the bot's core schema. See
``docs/llm-calls-schema.md`` for the canonical DDL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PREVIEW_LEN = 200

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    job TEXT NOT NULL,
    model TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    error TEXT,
    response_preview TEXT
);
"""


class LlmCallsLog:
    """Append-only log of LLM call attempts in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row  # enables row["col"] access
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def log_call(
        self,
        *,
        job: str,
        model: str,
        attempt: int,
        status: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        latency_ms: Optional[int],
        error: Optional[str] = None,
        response_preview: Optional[str] = None,
    ) -> int:
        """Insert one attempt row. Returns the row id."""
        if response_preview is not None:
            response_preview = response_preview[:PREVIEW_LEN]
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO llm_calls
                (created_at, job, model, attempt, status,
                 prompt_tokens, completion_tokens, total_tokens,
                 latency_ms, error, response_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at, job, model, attempt, status,
                prompt_tokens, completion_tokens, total_tokens,
                latency_ms, error, response_preview,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 10) -> list[sqlite3.Row]:
        """Return the most recent rows (newest first) for inspection."""
        cur = self._conn.execute(
            "SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM llm_calls")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LlmCallsLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()