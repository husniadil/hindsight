"""Shared utilities for Claude Agent SDK integration."""
from __future__ import annotations

import asyncio
import logging
import os

_DEFAULT_MAX_CONCURRENT = 3

logger = logging.getLogger(__name__)

_shared_semaphore: asyncio.Semaphore | None = None


def get_claude_sdk_semaphore(
    max_concurrent: int | None = None,
) -> asyncio.Semaphore:
    """Return a shared asyncio.Semaphore for Claude SDK concurrency control.

    Returns the same instance on repeated calls (singleton per process).
    Reads HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT env var if max_concurrent not provided.
    """
    global _shared_semaphore
    if max_concurrent is not None:
        # Explicit value — return a new semaphore (for testing)
        return asyncio.Semaphore(max_concurrent)
    if _shared_semaphore is None:
        value = int(os.getenv("HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT", str(_DEFAULT_MAX_CONCURRENT)))
        _shared_semaphore = asyncio.Semaphore(value)
    return _shared_semaphore


def log_sdk_messages(
    msg: object,
    *,
    agent_name: str,
    log: logging.Logger | None = None,
) -> None:
    """Log Claude Agent SDK messages in a concise format.

    Call this inside the ``async for msg in client.receive_messages()`` loop.
    Handles AssistantMessage (text/tool_use blocks) and ResultMessage.
    """
    _log = log or logger

    # Lazy imports to avoid top-level SDK dependency
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                # Truncate large input for readability
                input_repr = str(block.input)
                if len(input_repr) > 200:
                    input_repr = input_repr[:200] + "..."
                _log.info(
                    "[%s] tool_use: %s | input: %s",
                    agent_name,
                    block.name,
                    input_repr,
                )
            elif isinstance(block, TextBlock):
                preview = block.text[:150].replace("\n", " ")
                if len(block.text) > 150:
                    preview += "..."
                _log.debug("[%s] text: %s", agent_name, preview)

    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "n/a"
        _log.info(
            "[%s] done | turns=%s duration=%sms cost=%s session=%s",
            agent_name,
            msg.num_turns,
            msg.duration_ms,
            cost,
            msg.session_id,
        )
