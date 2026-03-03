"""Shared utilities for Claude Agent SDK integration."""
from __future__ import annotations

import asyncio
import os

_DEFAULT_MAX_CONCURRENT = 3

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
