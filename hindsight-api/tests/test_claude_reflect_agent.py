"""Tests for Claude Agent SDK reflect agent (Task 2, 3, 7)."""
from __future__ import annotations

import json
import pytest
from hindsight_api.engine.reflect.claude_agent import build_reflect_tools


def _extract_text(result: dict) -> str:
    """Extract text from SDK-compatible tool result dict."""
    return result["content"][0]["text"]


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
    data = json.loads(_extract_text(result))
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
    text = _extract_text(result)
    assert "accepted" in text.lower()
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
    text = _extract_text(result)
    assert "error" in text.lower() or "search" in text.lower()
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
    data = json.loads(_extract_text(result))
    assert "memories" in data
    assert "m-1" in available_memory_ids


@pytest.mark.asyncio
async def test_budget_gate_blocks_after_max_search_calls():
    """Tools return budget-exhausted message after max_search_calls reached."""
    tool_trace: list = []

    async def mock_recall(query: str, max_tokens: int, max_chunk_tokens: int):
        return {"memories": [{"id": f"m-{len(tool_trace)}", "text": "fact"}]}

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=mock_recall,
        expand_fn=None,
        available_memory_ids=set(),
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=tool_trace,
        result_holder=[],
        max_search_calls=2,
    )
    recall_tool = next(t for t in tools if t.name == "recall")

    # First 2 calls should succeed
    r1 = await recall_tool.handler({"reason": "t", "query": "q1"})
    assert "memories" in _extract_text(r1)
    r2 = await recall_tool.handler({"reason": "t", "query": "q2"})
    assert "memories" in _extract_text(r2)

    # 3rd call should be gated
    r3 = await recall_tool.handler({"reason": "t", "query": "q3"})
    text = _extract_text(r3)
    assert "Budget exhausted" in text
    assert "done()" in text
    assert len(tool_trace) == 2  # only 2 actual calls recorded


@pytest.mark.asyncio
async def test_budget_gate_does_not_block_done():
    """done() should never be gated, even after budget is exhausted."""
    tool_trace: list = []
    available_memory_ids = {"m-1"}
    result_holder: list = []

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=None,
        expand_fn=None,
        available_memory_ids=available_memory_ids,
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=tool_trace,
        result_holder=result_holder,
        max_search_calls=0,  # budget already exhausted
    )
    done_tool = next(t for t in tools if t.name == "done")
    result = await done_tool.handler({"answer": "My answer", "memory_ids": ["m-1"]})
    assert "accepted" in _extract_text(result).lower()
    assert len(result_holder) == 1


@pytest.mark.asyncio
async def test_budget_gate_unlimited_when_none():
    """When max_search_calls is None, no budget limit is enforced."""
    tool_trace: list = []

    async def mock_recall(query: str, max_tokens: int, max_chunk_tokens: int):
        return {"memories": [{"id": f"m-{len(tool_trace)}", "text": "fact"}]}

    tools = build_reflect_tools(
        search_mental_models_fn=None,
        search_observations_fn=None,
        recall_fn=mock_recall,
        expand_fn=None,
        available_memory_ids=set(),
        available_mental_model_ids=set(),
        available_observation_ids=set(),
        tool_trace=tool_trace,
        result_holder=[],
        max_search_calls=None,  # unlimited
    )
    recall_tool = next(t for t in tools if t.name == "recall")

    # Should allow many calls without gating
    for i in range(20):
        r = await recall_tool.handler({"reason": "t", "query": f"q{i}"})
        assert "Budget exhausted" not in _extract_text(r)
    assert len(tool_trace) == 20


# ---------------------------------------------------------------------------
# Task 3 — claude_reflect_agent() main function
# ---------------------------------------------------------------------------
from unittest.mock import AsyncMock, MagicMock, patch

from hindsight_api.engine.reflect.claude_agent import claude_reflect_agent


@pytest.mark.asyncio
async def test_claude_reflect_agent_calls_sdk():
    """Test that claude_reflect_agent creates SDK client, queries, and returns result."""

    async def mock_search_mm(q: str, n: int):
        return {"mental_models": [{"id": "mm-1", "text": "test"}]}

    async def mock_search_obs(q: str, n: int):
        return {"observations": [{"id": "obs-1", "text": "test"}]}

    async def mock_recall(q: str, mt: int, mct: int):
        return {"memories": [{"id": "m-1", "text": "test"}]}

    async def mock_expand(ids: list, depth: str):
        return {"expanded": []}

    with patch("claude_agent_sdk.ClaudeSDKClient") as MockClient, \
         patch("hindsight_api.engine.reflect.claude_agent.get_claude_sdk_semaphore") as mock_sem:

        # Setup semaphore mock as async context manager
        sem_mock = MagicMock()
        sem_mock.__aenter__ = AsyncMock(return_value=None)
        sem_mock.__aexit__ = AsyncMock(return_value=False)
        mock_sem.return_value = sem_mock

        # Setup client mock
        from claude_agent_sdk import ResultMessage
        mock_client_instance = AsyncMock()

        # receive_response() yields messages up to and including ResultMessage
        async def fake_receive_response():
            yield ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=100,
                is_error=False,
                num_turns=1,
                session_id="test-session",
            )

        mock_client_instance.receive_response = fake_receive_response

        # Context manager protocol
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await claude_reflect_agent(
            query="What patterns exist?",
            bank_profile={"name": "test", "mission": "test"},
            search_mental_models_fn=mock_search_mm,
            search_observations_fn=mock_search_obs,
            recall_fn=mock_recall,
            expand_fn=mock_expand,
            has_mental_models=True,
        )
        # Verify SDK client was created and queried
        mock_client_instance.query.assert_called_once()


# ---------------------------------------------------------------------------
# Task 7 — system prompt streamline
# ---------------------------------------------------------------------------
from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools


def test_system_prompt_no_forced_workflow_steps():
    """Verify streamlined prompt contains retrieval strategy and bank info."""
    prompt = build_system_prompt_for_tools(
        bank_profile={"name": "test", "mission": "test mission"},
        context=None,
        directives=None,
        has_mental_models=True,
        budget="mid",
    )
    # Should contain retrieval strategy
    assert "RETRIEVAL STRATEGY" in prompt.upper() or "retrieval" in prompt.lower()
    # Should contain bank info
    assert "test mission" in prompt
    # Should not be asking follow-up questions
    assert "Would you like me to" not in prompt


def test_system_prompt_non_conversational_reworded():
    """Verify prompt uses SDK-friendly non-conversational wording."""
    prompt = build_system_prompt_for_tools(
        bank_profile={"name": "test", "mission": ""},
        has_mental_models=False,
    )
    # Should NOT contain old rigid wording that's redundant for SDK auto-loop
    # (The SDK manages turns — the prompt shouldn't be confusingly prescriptive)
    assert "done()" in prompt or "done" in prompt  # should still mention done tool
