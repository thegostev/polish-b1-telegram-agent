#!/usr/bin/env python3
"""Polish correction smoke test — the FYR-187 "done when" deliverable.

Takes a Polish sentence, returns a model correction via the config-chosen model, and
logs the call to the ``llm_calls`` SQLite log.

Examples:
    .venv/bin/python scripts/correct.py "Ja jest w domu"
    echo "Ja jest w domu" | .venv/bin/python scripts/correct.py --stdin
    .venv/bin/python scripts/correct.py "Ja jest w domu" --model glm-5.2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run as `python scripts/correct.py` without installing: put repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from babciobot.config import load_config
from babciobot.llm import Correction, LLMClient

# Ask for a Polish correction. Polish prompt keeps the model in-language.
SYSTEM_PROMPT = (
    "Jesteś korektorem języka polskiego. Popraw podane zdanie użytkownika. "
    "Zwróć poprawione zdanie i krótkie wyjaśnienie błędu po polsku."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct a Polish sentence via the configured LLM model (FYR-187 smoke test)."
    )
    parser.add_argument("sentence", nargs="?", help="Polish sentence to correct (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read the sentence from stdin")
    parser.add_argument(
        "--job", default="correction", help="Configured job name whose model to use (default: correction)"
    )
    parser.add_argument("--model", default=None, help="Override the configured model for this job")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml (default: config.toml)")
    args = parser.parse_args()

    if args.stdin:
        sentence = sys.stdin.read().strip()
    elif args.sentence:
        sentence = args.sentence
    else:
        parser.error("Provide a sentence argument or use --stdin.")

    config = load_config(args.config)
    client = LLMClient(config)
    try:
        result = client.complete_json(
            args.job,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            Correction,
            model=args.model,
        )
    finally:
        client.close()

    used_model = args.model or config.models.get(args.job, "?")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] job={args.job} model={used_model}")
    print(f"  input       : {sentence}")
    print(f"  corrected   : {result.corrected}")
    print(f"  explanation : {result.explanation}")
    print(f"  logged to   : {config.storage.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())