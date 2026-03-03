"""Tests for claude_consolidate_agent — build_consolidate_tool."""
from __future__ import annotations

import pytest
from hindsight_api.engine.consolidation.claude_agent import build_consolidate_tool


@pytest.mark.asyncio
async def test_consolidate_handler():
    result_holder: list[dict] = []

    tool = build_consolidate_tool(result_holder=result_holder)
    result = await tool.handler({
        "creates": [{"text": "New observation", "source_fact_ids": ["f-1"]}],
        "updates": [{"observation_id": "obs-1", "text": "Updated", "source_fact_ids": ["f-2"]}],
        "deletes": [{"observation_id": "obs-2"}],
    })
    assert "accepted" in result.lower()
    assert len(result_holder) == 1
    assert len(result_holder[0]["creates"]) == 1
    assert len(result_holder[0]["updates"]) == 1
    assert len(result_holder[0]["deletes"]) == 1


@pytest.mark.asyncio
async def test_consolidate_handler_empty():
    result_holder: list[dict] = []

    tool = build_consolidate_tool(result_holder=result_holder)
    result = await tool.handler({"creates": [], "updates": [], "deletes": []})
    assert "accepted" in result.lower()
    assert len(result_holder) == 1
    assert result_holder[0]["creates"] == []
    assert result_holder[0]["updates"] == []
    assert result_holder[0]["deletes"] == []


@pytest.mark.asyncio
async def test_consolidate_handler_missing_keys():
    """Handler should tolerate missing keys with defaults."""
    result_holder: list[dict] = []

    tool = build_consolidate_tool(result_holder=result_holder)
    result = await tool.handler({})
    assert "accepted" in result.lower()
    assert len(result_holder) == 1
    assert result_holder[0]["creates"] == []
    assert result_holder[0]["updates"] == []
    assert result_holder[0]["deletes"] == []
