"""Tests for Claude Agent SDK reflect agent (Task 2, 3, 7)."""
from __future__ import annotations

import json
import pytest
from hindsight_api.engine.reflect.claude_agent import build_reflect_tools


@pytest.mark.asyncio
async def test_search_mental_models_handler():
    available_mental_model_ids: set[str] = set()
    tool_trace: list = []

    async def mock_search_fn(query: str, max_results: int):
        return {"mental_models": [{"id": "mm-1", "text": "pattern A"}]}

    tools = build_reflect_tools(
        search_mental_models_fn=mock_search_fn,
        search_observations_fn=None,
        recall_fn=None,
        expand_fn=None,
        available_memory_ids=set(),
        available_mental_model_ids=available_mental_model_ids,
        available_observation_ids=set(),
        tool_trace=tool_trace,
        result_holder=[],
    )
    smm_tool = next(t for t in tools if t.name == "search_mental_models")
    result = await smm_tool.handler({"reason": "test", "query": "patterns", "max_results": 3})
    data = json.loads(result)
    assert data["mental_models"][0]["id"] == "mm-1"
    assert "mm-1" in available_mental_model_ids
    assert len(tool_trace) == 1


@pytest.mark.asyncio
async def test_done_handler_validates_ids():
    available_memory_ids = {"m-1", "m-2"}
    result_holder: list = []

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=None,
        expand_fn=None,
        available_memory_ids=available_memory_ids,
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=[],
        result_holder=result_holder,
    )
    done_tool = next(t for t in tools if t.name == "done")
    result = await done_tool.handler({
        "answer": "The answer is X.",
        "memory_ids": ["m-1", "m-999"],  # m-999 not in available
    })
    assert "accepted" in result.lower()
    assert len(result_holder) == 1
    assert "m-1" in result_holder[0].used_memory_ids
    assert "m-999" not in result_holder[0].used_memory_ids


@pytest.mark.asyncio
async def test_done_handler_blocks_without_evidence():
    result_holder: list = []

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=None,
        expand_fn=None,
        available_memory_ids=set(),          # empty — no evidence
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=[],
        result_holder=result_holder,
    )
    done_tool = next(t for t in tools if t.name == "done")
    result = await done_tool.handler({"answer": "No evidence answer"})
    assert "error" in result.lower() or "search" in result.lower()
    assert len(result_holder) == 0  # not stored


@pytest.mark.asyncio
async def test_recall_handler():
    available_memory_ids: set[str] = set()

    async def mock_recall(query: str, max_tokens: int, max_chunk_tokens: int):
        return {"memories": [{"id": "m-1", "text": "fact A"}]}

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=mock_recall,
        expand_fn=None,
        available_memory_ids=available_memory_ids,
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=[],
        result_holder=[],
    )
    recall_tool = next(t for t in tools if t.name == "recall")
    result = await recall_tool.handler({"reason": "need facts", "query": "test"})
    data = json.loads(result)
    assert "memories" in data
    assert "m-1" in available_memory_ids
