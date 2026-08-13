"""babciobot — Telegram bot coaching Alex and Olga toward a Polish B1 certificate.

This package is being built up ticket by ticket. FYR-187 adds only the LLM client
layer (config-driven models, one retry/timeout policy, an ``llm_calls`` SQLite log).
The Telegram bot skeleton, full core schema, and launchd service come later (T-03/T-06).
"""

__version__ = "0.0.1"