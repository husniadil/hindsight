"""Shared utilities for Claude Agent SDK integration."""
from __future__ import annotations

import asyncio
import os

_DEFAULT_MAX_CONCURRENT = 3


def get_claude_sdk_semaphore(
    max_concurrent: int | None = None,
) -> asyncio.Semaphore:
    """Return an asyncio.Semaphore for Claude SDK concurrency control.

    Reads HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT env var if max_concurrent not provided.
    """
    if max_concurrent is None:
        max_concurrent = int(
            os.getenv("HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT", str(_DEFAULT_MAX_CONCURRENT))
        )
    return asyncio.Semaphore(max_concurrent)
