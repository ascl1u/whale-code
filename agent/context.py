"""Context management utilities."""

from __future__ import annotations

import copy
from typing import Any


def add_anthropic_caching(
    messages: list[dict[str, Any]], model_name: str
) -> list[dict[str, Any]]:
    """Add ephemeral cache_control to the last 3 messages for Anthropic models.

    This enables Anthropic's prompt caching, which reduces cost and latency
    for the shared prefix of the conversation history.
    """
    if not ("anthropic" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    cached = copy.deepcopy(messages)
    for i in range(max(0, len(cached) - 3), len(cached)):
        msg = cached[i]
        if isinstance(msg.get("content"), str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and "type" in item:
                    item["cache_control"] = {"type": "ephemeral"}
    return cached
