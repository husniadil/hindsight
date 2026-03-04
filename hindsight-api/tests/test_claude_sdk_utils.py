"""Tests for Claude SDK shared utilities."""
from __future__ import annotations

import asyncio
import logging

import pytest

from hindsight_api.engine import claude_sdk_utils
from hindsight_api.engine.claude_sdk_utils import get_claude_sdk_semaphore, log_sdk_messages


@pytest.fixture(autouse=True)
def reset_shared_semaphore():
    """Reset the shared singleton between tests."""
    claude_sdk_utils._shared_semaphore = None
    yield
    claude_sdk_utils._shared_semaphore = None


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    sem = get_claude_sdk_semaphore(max_concurrent=2)
    order: list[str] = []

    async def task(name: str):
        async with sem:
            order.append(f"{name}_start")
            await asyncio.sleep(0.05)
            order.append(f"{name}_end")

    await asyncio.gather(task("a"), task("b"), task("c"))
    # c must start after a or b ends (only 2 concurrent allowed)
    c_start_idx = order.index("c_start")
    assert any(order.index(f"{x}_end") < c_start_idx for x in ("a", "b"))


@pytest.mark.asyncio
async def test_semaphore_default_value():
    sem = get_claude_sdk_semaphore()
    assert isinstance(sem, asyncio.Semaphore)
    assert sem._value == 3  # default


@pytest.mark.asyncio
async def test_semaphore_from_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT", "5")
    sem = get_claude_sdk_semaphore()
    assert isinstance(sem, asyncio.Semaphore)
    assert sem._value == 5


@pytest.mark.asyncio
async def test_semaphore_is_singleton():
    """Repeated calls without explicit max_concurrent return the same instance."""
    sem1 = get_claude_sdk_semaphore()
    sem2 = get_claude_sdk_semaphore()
    assert sem1 is sem2


def test_log_sdk_messages_tool_use(caplog):
    """log_sdk_messages logs ToolUseBlock at INFO level."""
    from claude_agent_sdk import AssistantMessage, ToolUseBlock

    msg = AssistantMessage(
        content=[ToolUseBlock(id="t-1", name="recall", input={"query": "test"})],
        model="test",
    )
    with caplog.at_level(logging.INFO):
        log_sdk_messages(msg, agent_name="reflect")
    assert "[reflect] tool_use: recall" in caplog.text


def test_log_sdk_messages_result(caplog):
    """log_sdk_messages logs ResultMessage summary."""
    from claude_agent_sdk import ResultMessage

    msg = ResultMessage(
        subtype="success",
        duration_ms=500,
        duration_api_ms=400,
        is_error=False,
        num_turns=3,
        session_id="sess-123",
    )
    with caplog.at_level(logging.INFO):
        log_sdk_messages(msg, agent_name="retain")
    assert "[retain] done" in caplog.text
    assert "turns=3" in caplog.text
    assert "sess-123" in caplog.text


def test_log_sdk_messages_text_at_debug(caplog):
    """TextBlock is logged at DEBUG, not shown at INFO."""
    from claude_agent_sdk import AssistantMessage, TextBlock

    msg = AssistantMessage(
        content=[TextBlock(text="Some analysis text here")],
        model="test",
    )
    with caplog.at_level(logging.INFO):
        log_sdk_messages(msg, agent_name="consolidation")
    assert caplog.text == ""  # nothing at INFO

    with caplog.at_level(logging.DEBUG):
        log_sdk_messages(msg, agent_name="consolidation")
    assert "[consolidation] text:" in caplog.text


def test_log_sdk_messages_unknown_type():
    """Unknown message types are silently ignored."""
    log_sdk_messages("not a real message", agent_name="test")  # no error
