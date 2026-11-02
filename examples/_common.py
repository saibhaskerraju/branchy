"""Shared helpers for the examples: LLM settings from env, offline stub."""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Six canned turns across two topics, used by the offline runs.
SAMPLE_TURNS = [
    ("How do I build an index in Postgres?", "Use CREATE INDEX idx ON table(col); btree by default."),
    ("Give me a vegan pasta sauce recipe", "Creamy tomato-basil: onion, garlic, tomatoes, cashews, basil."),
    ("What if I need it to not lock the table?", "Use CREATE INDEX CONCURRENTLY; it cannot run inside a transaction."),
    ("I don't like onion, can you change it?", "No-onion version: extra garlic, fennel seed, tomatoes, cashews."),
    ("How much salt?", "Half a teaspoon when adding tomatoes, adjust after simmering."),
    ("Can that run inside a migration transaction?", "No - CONCURRENTLY cannot run in a transaction block."),
]


def llm_kwargs() -> dict:
    """model/api_key/api_base for BranchingChat from LLM_* (or OPENAI_API_KEY)."""
    model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("set LLM_MODEL/LLM_API_KEY (or OPENAI_API_KEY), or use --offline")
    return {"model": model, "api_key": api_key, "api_base": os.environ.get("LLM_API_BASE")}


_CANNED = {u.lower(): a for u, a in SAMPLE_TURNS}


def stub_responder(history: list[dict], user_message: str) -> str:
    """Deterministic stand-in for an LLM. Known sample questions get their
    canned answer (so the router has real text to match on); anything else
    gets an echo that says how much history was received."""
    canned = _CANNED.get(user_message.strip().lower())
    if canned:
        return canned
    return f"(stub) {len(history)} prior turn(s) seen; you said: {user_message}"


def describe(decision) -> str:
    """One-line routing summary for printing."""
    if decision.action == "NEW":
        return "[NEW thread]"
    return f"[{decision.action} -> {decision.target_message_id[:8]}] {decision.reason}"
