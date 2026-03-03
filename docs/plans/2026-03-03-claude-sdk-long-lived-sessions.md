# Claude Agent SDK Long-lived Sessions — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate Hindsight's claude-code provider from stateless one-shot sessions to long-lived sessions with real MCP tool handlers and SDK-managed multi-turn conversations.

**Architecture:** When provider is `claude-code`, reflect/retain/consolidation bypass `LLMProvider` and use dedicated `claude_*_agent()` modules with `ClaudeSDKClient` per call, real MCP tool handlers as closures, and a dedicated concurrency semaphore. All non-claude-code providers remain unchanged.

**Tech Stack:** `claude-agent-sdk>=0.1.27` (`ClaudeSDKClient`, `create_sdk_mcp_server`, `SdkMcpTool`), Python 3.12+, pytest with `asyncio_mode=auto`

**Design doc:** `docs/plans/2026-03-03-claude-sdk-long-lived-sessions-design.md`

**Base path:** `hindsight-api/hindsight_api/engine/`
**Test path:** `hindsight-api/tests/`

## Execution Strategy: Agent Team

Use `TeamCreate` to create a team and dispatch tasks in parallel where possible.

### Team Composition

| Agent Name | Type | Tasks | Rationale |
|------------|------|-------|-----------|
| `lead` | Team lead (you) | Task 6 (integration), Task 8 (regression), coordination | Integration wiring touches `memory_engine.py` — central file, needs to see all agents' output |
| `reflect-dev` | `general-purpose` | Task 2, Task 3, Task 7 | Reflect agent: tool builders → main function → prompt streamline (sequential dependency) |
| `retain-dev` | `general-purpose` | Task 4 | Retain agent: independent from reflect |
| `consolidation-dev` | `general-purpose` | Task 5 | Consolidation agent: independent from reflect and retain |

### Parallel Execution Plan

```
Phase 0: Lead creates Task 1 (semaphore utility) — shared dependency
         ↓
Phase 1: All 3 agents start in parallel (after Task 1 committed):
         reflect-dev  → Task 2 → Task 3 → Task 7
         retain-dev   → Task 4
         consolidation-dev → Task 5
         ↓
Phase 2: Lead does Task 6 (integration wiring) after all agents complete
         ↓
Phase 3: Lead does Task 8 (full regression test)
```

### Agent Instructions Template

Each agent receives:
- This plan file path for context
- Their assigned task numbers
- Working directory: `/Users/husni/.genduk/mcp-servers/hindsight/hindsight-api`
- Instruction to commit after each task
- Instruction to notify lead when done

### Worktree Strategy

All agents work in **the same worktree** (not isolated) because:
- Task 1 (semaphore) is a shared dependency committed before agents start
- Each agent creates files in different directories (no conflicts)
- Integration (Task 6) needs to see all new files

---

## Task 1: Shared Concurrency Semaphore

**Files:**
- Create: `engine/claude_sdk_utils.py`
- Test: `tests/test_claude_sdk_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_claude_sdk_utils.py
import asyncio
import pytest
from hindsight_api.engine.claude_sdk_utils import get_claude_sdk_semaphore


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    sem = get_claude_sdk_semaphore(max_concurrent=2)
    running = []
    order = []

    async def task(name: str):
        async with sem:
            running.append(name)
            order.append(f"{name}_start")
            await asyncio.sleep(0.05)
            order.append(f"{name}_end")
            running.remove(name)

    await asyncio.gather(task("a"), task("b"), task("c"))
    # At no point should more than 2 be running
    assert "c_start" in order
    # c must start after a or b ends
    c_start_idx = order.index("c_start")
    assert any(order.index(f"{x}_end") < c_start_idx for x in ("a", "b"))


@pytest.mark.asyncio
async def test_semaphore_default_value():
    sem = get_claude_sdk_semaphore()
    assert isinstance(sem, asyncio.Semaphore)
```

**Step 2: Run test to verify it fails**

Run: `cd hindsight-api && python -m pytest tests/test_claude_sdk_utils.py -v --timeout 30 -n 0`
Expected: FAIL — `ModuleNotFoundError: No module named 'hindsight_api.engine.claude_sdk_utils'`

**Step 3: Write minimal implementation**

```python
# engine/claude_sdk_utils.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd hindsight-api && python -m pytest tests/test_claude_sdk_utils.py -v --timeout 30 -n 0`
Expected: PASS

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/claude_sdk_utils.py tests/test_claude_sdk_utils.py
git commit -m "feat: add Claude SDK concurrency semaphore utility"
```

---

## Task 2: Reflect Agent — Real Tool Builders

**Files:**
- Create: `engine/reflect/claude_agent.py`
- Test: `tests/test_claude_reflect_agent.py`

**Step 1: Write the failing test — tool handlers**

```python
# tests/test_claude_reflect_agent.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd hindsight-api && python -m pytest tests/test_claude_reflect_agent.py -v --timeout 30 -n 0`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# engine/reflect/claude_agent.py
"""Claude Agent SDK reflect agent with real MCP tool handlers."""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from claude_agent_sdk import SdkMcpTool

from ..reflect.agent import ReflectAgentResult, ToolCall

logger = logging.getLogger(__name__)


def _extract_ids_from_results(data: dict, key: str, id_field: str = "id") -> set[str]:
    """Extract IDs from tool result dict."""
    ids: set[str] = set()
    items = data.get(key, [])
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and id_field in item:
                ids.add(str(item[id_field]))
    return ids


def build_reflect_tools(
    *,
    search_mental_models_fn: Callable | None,
    search_observations_fn: Callable | None,
    recall_fn: Callable | None,
    expand_fn: Callable | None,
    available_memory_ids: set[str],
    available_mental_model_ids: set[str],
    available_observation_ids: set[str],
    tool_trace: list[ToolCall],
    result_holder: list[ReflectAgentResult],
) -> list[SdkMcpTool]:
    """Build real MCP tool handlers as closures capturing shared mutable state."""

    async def search_mental_models_handler(args: dict) -> str:
        query = str(args.get("query", ""))
        max_results = int(args.get("max_results", 5))
        try:
            data = await search_mental_models_fn(query, max_results)
            ids = _extract_ids_from_results(data, "mental_models")
            available_mental_model_ids.update(ids)
            tool_trace.append(ToolCall(tool="search_mental_models", query=query, output=data))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"search_mental_models error: {exc}")
            return json.dumps({"error": str(exc)})

    async def search_observations_handler(args: dict) -> str:
        query = str(args.get("query", ""))
        max_tokens = int(args.get("max_tokens", 5000))
        try:
            data = await search_observations_fn(query, max_tokens)
            ids = _extract_ids_from_results(data, "observations")
            available_observation_ids.update(ids)
            tool_trace.append(ToolCall(tool="search_observations", query=query, output=data))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"search_observations error: {exc}")
            return json.dumps({"error": str(exc)})

    async def recall_handler(args: dict) -> str:
        query = str(args.get("query", ""))
        max_tokens = int(args.get("max_tokens", 2048))
        max_chunk_tokens = int(args.get("max_chunk_tokens", 1000))
        try:
            data = await recall_fn(query, max_tokens, max_chunk_tokens)
            ids = _extract_ids_from_results(data, "memories")
            available_memory_ids.update(ids)
            tool_trace.append(ToolCall(tool="recall", query=query, output=data))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"recall error: {exc}")
            return json.dumps({"error": str(exc)})

    async def expand_handler(args: dict) -> str:
        memory_ids = args.get("memory_ids", [])
        depth = str(args.get("depth", "chunk"))
        try:
            data = await expand_fn(memory_ids, depth)
            tool_trace.append(ToolCall(tool="expand", query=str(memory_ids), output=data))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"expand error: {exc}")
            return json.dumps({"error": str(exc)})

    async def done_handler(args: dict) -> str:
        # Block if no evidence gathered
        if not available_memory_ids and not available_mental_model_ids and not available_observation_ids:
            return json.dumps({
                "error": "You must search for information first. "
                "Use search_mental_models(), search_observations(), or recall() before providing your final answer."
            })

        answer = str(args.get("answer", ""))
        # Validate cited IDs against available sets
        cited_memory_ids = set(args.get("memory_ids", []))
        cited_mental_model_ids = set(args.get("mental_model_ids", []))
        cited_observation_ids = set(args.get("observation_ids", []))

        result = ReflectAgentResult(
            answer=answer,
            used_memory_ids=cited_memory_ids & available_memory_ids,
            used_mental_model_ids=cited_mental_model_ids & available_mental_model_ids,
            used_observation_ids=cited_observation_ids & available_observation_ids,
            tool_trace=list(tool_trace),
        )
        result_holder.append(result)
        return "Answer accepted."

    # Import tool schemas from existing module
    from ..reflect.tools_schema import get_reflect_tool_schemas

    schemas = get_reflect_tool_schemas()

    tools = []
    handler_map = {
        "search_mental_models": search_mental_models_handler,
        "search_observations": search_observations_handler,
        "recall": recall_handler,
        "expand": expand_handler,
        "done": done_handler,
    }
    for schema in schemas:
        name = schema["function"]["name"]
        if name in handler_map:
            tools.append(SdkMcpTool(
                name=name,
                description=schema["function"]["description"],
                input_schema=schema["function"]["parameters"],
                handler=handler_map[name],
            ))
    return tools
```

> **Note:** `ToolCall`, `ReflectAgentResult`, and `get_reflect_tool_schemas` need to be importable from existing modules. `get_reflect_tool_schemas` is a thin wrapper around `get_reflect_tools()` from `tools_schema.py` — may need a small adapter if the existing function returns OpenAI format. Verify during implementation and adapt.

**Step 4: Run test to verify it passes**

Run: `cd hindsight-api && python -m pytest tests/test_claude_reflect_agent.py -v --timeout 30 -n 0`
Expected: PASS (may need import adjustments — `ToolCall` and `ReflectAgentResult` dataclass imports)

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/reflect/claude_agent.py tests/test_claude_reflect_agent.py
git commit -m "feat: add real MCP tool handlers for Claude reflect agent"
```

---

## Task 3: Reflect Agent — `claude_reflect_agent()` Main Function

**Files:**
- Modify: `engine/reflect/claude_agent.py`
- Test: `tests/test_claude_reflect_agent.py` (add tests)

**Step 1: Write the failing test**

```python
# Append to tests/test_claude_reflect_agent.py
from unittest.mock import AsyncMock, MagicMock, patch
from hindsight_api.engine.reflect.claude_agent import claude_reflect_agent


@pytest.mark.asyncio
async def test_claude_reflect_agent_calls_sdk():
    """Test that claude_reflect_agent creates SDK client, queries, and returns result."""
    mock_result_message = MagicMock()
    mock_result_message.__class__.__name__ = "ResultMessage"

    async def mock_search_mm(q, n):
        return {"mental_models": [{"id": "mm-1", "text": "test"}]}

    async def mock_search_obs(q, n):
        return {"observations": [{"id": "obs-1", "text": "test"}]}

    async def mock_recall(q, mt, mct):
        return {"memories": [{"id": "m-1", "text": "test"}]}

    async def mock_expand(ids, depth):
        return {"expanded": []}

    with patch("hindsight_api.engine.reflect.claude_agent.ClaudeSDKClient") as MockClient, \
         patch("hindsight_api.engine.reflect.claude_agent.get_claude_sdk_semaphore") as mock_sem:

        # Setup semaphore mock
        mock_sem.return_value = asyncio.Semaphore(1)

        # Setup client mock — simulate SDK calling tools then returning
        mock_client_instance = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        # Simulate: SDK auto-loops tools, then ResultMessage
        from claude_agent_sdk import ResultMessage
        mock_client_instance.receive_messages.return_value = AsyncMock()
        mock_client_instance.receive_messages.return_value.__aiter__ = AsyncMock(
            return_value=iter([ResultMessage(session_id="test-session")])
        )

        # We need to simulate the done handler being called by SDK
        # In real SDK, tools are called internally — for unit test, we pre-populate result_holder
        # Integration test would verify full flow

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
```

**Step 2: Run test to verify it fails**

Run: `cd hindsight-api && python -m pytest tests/test_claude_reflect_agent.py::test_claude_reflect_agent_calls_sdk -v --timeout 30 -n 0`
Expected: FAIL — `ImportError: cannot import name 'claude_reflect_agent'`

**Step 3: Write implementation**

```python
# Append to engine/reflect/claude_agent.py

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    AssistantMessage,
    create_sdk_mcp_server,
)

from ..claude_sdk_utils import get_claude_sdk_semaphore
from .prompts import build_system_prompt_for_tools

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = get_claude_sdk_semaphore()
    return _semaphore


async def claude_reflect_agent(
    query: str,
    bank_profile: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    context: str | None = None,
    max_iterations: int = 10,
    max_tokens: int | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    budget: str | None = None,
) -> ReflectAgentResult:
    """Run reflect agent using Claude SDK with real MCP tool handlers.

    Creates a ClaudeSDKClient per call. SDK auto-loops tool calls until done() is called.
    """
    import asyncio

    # Shared mutable state for closures
    available_memory_ids: set[str] = set()
    available_mental_model_ids: set[str] = set()
    available_observation_ids: set[str] = set()
    tool_trace: list[ToolCall] = []
    result_holder: list[ReflectAgentResult] = []

    # Build real tool handlers
    sdk_tools = build_reflect_tools(
        search_mental_models_fn=search_mental_models_fn,
        search_observations_fn=search_observations_fn,
        recall_fn=recall_fn,
        expand_fn=expand_fn,
        available_memory_ids=available_memory_ids,
        available_mental_model_ids=available_mental_model_ids,
        available_observation_ids=available_observation_ids,
        tool_trace=tool_trace,
        result_holder=result_holder,
    )

    # Create in-process MCP server
    mcp_server = create_sdk_mcp_server(
        name="hindsight",
        version="1.0.0",
        tools=sdk_tools,
    )

    # Build system prompt (reuse existing, streamlined)
    system_prompt = build_system_prompt_for_tools(
        bank_profile=bank_profile,
        context=context,
        directives=directives,
        has_mental_models=has_mental_models,
        budget=budget,
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=max_iterations + 2,  # buffer for done()
        mcp_servers={"hindsight": mcp_server},
        allowed_tools=[f"mcp__hindsight__{t.name}" for t in sdk_tools],
    )

    fallback_text = ""

    async with _get_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            await client.query(query)
            async for msg in client.receive_messages():
                if isinstance(msg, AssistantMessage):
                    # Capture last assistant text as fallback
                    from claude_agent_sdk import TextBlock
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            fallback_text = block.text
                if isinstance(msg, ResultMessage):
                    break

    # Return result from done() handler, or fallback
    if result_holder:
        return result_holder[0]

    # Fallback: max_turns hit without done() — use last assistant text
    logger.warning("Claude reflect agent hit max_turns without calling done()")
    return ReflectAgentResult(
        answer=fallback_text.strip() if fallback_text else "I could not find relevant information.",
        used_memory_ids=available_memory_ids,
        used_mental_model_ids=available_mental_model_ids,
        used_observation_ids=available_observation_ids,
        tool_trace=tool_trace,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd hindsight-api && python -m pytest tests/test_claude_reflect_agent.py -v --timeout 30 -n 0`
Expected: PASS

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/reflect/claude_agent.py tests/test_claude_reflect_agent.py
git commit -m "feat: add claude_reflect_agent() main function with SDK auto-loop"
```

---

## Task 4: Retain Agent — `claude_retain_agent()`

**Files:**
- Create: `engine/retain/claude_agent.py`
- Test: `tests/test_claude_retain_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_claude_retain_agent.py
import json
import pytest
from hindsight_api.engine.retain.claude_agent import build_extract_facts_tool


@pytest.mark.asyncio
async def test_extract_facts_handler_lenient_parsing():
    all_facts: list = []
    chunk_metadata: list = []

    tool = build_extract_facts_tool(
        all_facts=all_facts,
        chunk_metadata=chunk_metadata,
    )
    # Simulate Claude calling with slightly imperfect data
    result = await tool.handler({
        "facts": [
            {"what": "John lives in NYC", "fact_type": "world"},
            {"what": "Alice is happy", "fact_type": "INVALID_TYPE"},  # auto-fix to "world"
            {"factual_core": "Fallback field name", "fact_type": "experience"},  # uses fallback
            {},  # skip — no "what"
        ]
    })
    assert len(all_facts) == 3  # 4th skipped
    assert all_facts[1].fact_type == "world"  # auto-fixed
    assert all_facts[2].fact == "Fallback field name"
    assert len(chunk_metadata) == 1


@pytest.mark.asyncio
async def test_extract_facts_handler_empty_facts():
    all_facts: list = []
    chunk_metadata: list = []

    tool = build_extract_facts_tool(all_facts=all_facts, chunk_metadata=chunk_metadata)
    result = await tool.handler({"facts": []})
    assert len(all_facts) == 0
    assert "0 facts" in result.lower() or "extracted 0" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd hindsight-api && python -m pytest tests/test_claude_retain_agent.py -v --timeout 30 -n 0`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# engine/retain/claude_agent.py
"""Claude Agent SDK retain agent with real extract_facts tool handler."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
)

from ..claude_sdk_utils import get_claude_sdk_semaphore
from .fact_extraction import Fact, chunk_text, _build_user_message

if TYPE_CHECKING:
    from .types import ChunkMetadata

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    import asyncio
    global _semaphore
    if _semaphore is None:
        _semaphore = get_claude_sdk_semaphore()
    return _semaphore


def build_extract_facts_tool(
    *,
    all_facts: list[Fact],
    chunk_metadata: list[tuple[str, int]],
) -> SdkMcpTool:
    """Build extract_facts tool with lenient handler."""

    async def handler(args: dict) -> str:
        raw_facts = args.get("facts", [])
        parsed: list[Fact] = []
        for fact_data in raw_facts:
            if not isinstance(fact_data, dict):
                continue
            what = fact_data.get("what") or fact_data.get("factual_core")
            if not what:
                continue
            fact_type = fact_data.get("fact_type", "world")
            if fact_type not in ("world", "experience", "opinion"):
                fact_type = "world"
            try:
                fact = Fact(fact=str(what), fact_type=fact_type)
                parsed.append(fact)
            except Exception:
                continue

        all_facts.extend(parsed)
        chunk_metadata.append(("chunk", len(parsed)))
        return f"Extracted {len(parsed)} facts."

    return SdkMcpTool(
        name="extract_facts",
        description="Submit extracted facts from the text chunk.",
        input_schema={
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "description": "List of extracted facts",
                    "items": {
                        "type": "object",
                        "properties": {
                            "what": {"type": "string", "description": "The factual statement"},
                            "fact_type": {"type": "string", "enum": ["world", "experience", "opinion"]},
                        },
                        "required": ["what"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["facts"],
        },
        handler=handler,
    )


async def claude_retain_agent(
    text: str,
    event_date: datetime | None,
    context: str,
    config: Any,
    metadata: dict[str, str] | None = None,
) -> tuple[list[Fact], list[tuple[str, int]], dict]:
    """Run retain using Claude SDK with sequential chunks in same session."""
    import asyncio

    chunks = chunk_text(text, max_chars=config.retain_chunk_size)
    all_facts: list[Fact] = []
    chunk_metadata: list[tuple[str, int]] = []

    tool = build_extract_facts_tool(all_facts=all_facts, chunk_metadata=chunk_metadata)
    mcp_server = create_sdk_mcp_server(name="hindsight_retain", version="1.0.0", tools=[tool])

    system_prompt = config.retain_system_prompt if hasattr(config, "retain_system_prompt") else (
        "You are a fact extraction system. Extract significant facts from text chunks. "
        "Call extract_facts with the extracted facts for each chunk."
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=2,  # per query: read chunk → call extract_facts
        mcp_servers={"hindsight_retain": mcp_server},
        allowed_tools=["mcp__hindsight_retain__extract_facts"],
    )

    async with _get_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            for i, chunk in enumerate(chunks):
                user_msg = _build_user_message(
                    chunk=chunk,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    event_date=event_date,
                    context=context,
                    metadata=metadata,
                )
                await client.query(user_msg)
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        break

    return all_facts, chunk_metadata, {}
```

**Step 4: Run test to verify it passes**

Run: `cd hindsight-api && python -m pytest tests/test_claude_retain_agent.py -v --timeout 30 -n 0`
Expected: PASS

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/retain/claude_agent.py tests/test_claude_retain_agent.py
git commit -m "feat: add claude_retain_agent() with sequential chunk processing"
```

---

## Task 5: Consolidation Agent — `claude_consolidate_agent()`

**Files:**
- Create: `engine/consolidation/claude_agent.py`
- Test: `tests/test_claude_consolidate_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_claude_consolidate_agent.py
import pytest
from hindsight_api.engine.consolidation.claude_agent import build_consolidate_tool


@pytest.mark.asyncio
async def test_consolidate_handler():
    result_holder: list = []

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
    result_holder: list = []

    tool = build_consolidate_tool(result_holder=result_holder)
    result = await tool.handler({"creates": [], "updates": [], "deletes": []})
    assert "accepted" in result.lower()
    assert len(result_holder) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd hindsight-api && python -m pytest tests/test_claude_consolidate_agent.py -v --timeout 30 -n 0`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# engine/consolidation/claude_agent.py
"""Claude Agent SDK consolidation agent with real consolidate tool handler."""
from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
)

from ..claude_sdk_utils import get_claude_sdk_semaphore
from .prompts import build_batch_consolidation_prompt

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    import asyncio
    global _semaphore
    if _semaphore is None:
        _semaphore = get_claude_sdk_semaphore()
    return _semaphore


def build_consolidate_tool(*, result_holder: list[dict]) -> SdkMcpTool:
    """Build consolidate tool handler."""

    async def handler(args: dict) -> str:
        result_holder.append({
            "creates": args.get("creates", []),
            "updates": args.get("updates", []),
            "deletes": args.get("deletes", []),
        })
        return "Consolidation accepted."

    return SdkMcpTool(
        name="consolidate",
        description="Submit consolidation results: create new observations, update existing ones, or delete obsolete ones.",
        input_schema={
            "type": "object",
            "properties": {
                "creates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_fact_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text", "source_fact_ids"],
                    },
                },
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {"type": "string"},
                            "text": {"type": "string"},
                            "source_fact_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["observation_id", "text"],
                    },
                },
                "deletes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {"type": "string"},
                        },
                        "required": ["observation_id"],
                    },
                },
            },
            "required": ["creates", "updates", "deletes"],
        },
        handler=handler,
    )


async def claude_consolidate_agent(
    memories: list[dict[str, Any]],
    observations_text: str,
    mission: str,
) -> dict:
    """Run consolidation using Claude SDK."""
    import asyncio

    result_holder: list[dict] = []
    tool = build_consolidate_tool(result_holder=result_holder)
    mcp_server = create_sdk_mcp_server(name="hindsight_consolidation", version="1.0.0", tools=[tool])

    system_prompt = build_batch_consolidation_prompt(mission).split("{facts_text}")[0]
    # Extract consolidation rules from the template as system prompt

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=2,
        mcp_servers={"hindsight_consolidation": mcp_server},
        allowed_tools=["mcp__hindsight_consolidation__consolidate"],
    )

    # Format user prompt with actual data
    facts_lines = "\n".join(
        f"[{m.get('id', '')}] {m.get('text', '')} | occurred_start={m.get('occurred_start', '')}"
        for m in memories
    )
    user_msg = f"## Facts\n{facts_lines}\n\n## Existing Observations\n{observations_text}"

    async with _get_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_msg)
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break

    if result_holder:
        return result_holder[0]
    return {"creates": [], "updates": [], "deletes": []}
```

**Step 4: Run test to verify it passes**

Run: `cd hindsight-api && python -m pytest tests/test_claude_consolidate_agent.py -v --timeout 30 -n 0`
Expected: PASS

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/consolidation/claude_agent.py tests/test_claude_consolidate_agent.py
git commit -m "feat: add claude_consolidate_agent() with consolidate tool"
```

---

## Task 6: Integration — Wire into `memory_engine.py`

**Files:**
- Modify: `engine/memory_engine.py` (lines ~4680, ~1936, consolidator call)
- Test: `tests/test_claude_sdk_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_claude_sdk_integration.py
import pytest
from unittest.mock import patch, MagicMock


def test_reflect_async_routes_to_claude_agent_when_claude_code():
    """Verify that reflect_async() calls claude_reflect_agent when provider is claude-code."""
    # This is a routing test — mock both paths, verify correct one is called
    with patch("hindsight_api.engine.memory_engine.run_reflect_agent") as mock_existing, \
         patch("hindsight_api.engine.reflect.claude_agent.claude_reflect_agent") as mock_claude:

        mock_llm_config = MagicMock()
        mock_llm_config.provider = "claude-code"

        # Verify the import path exists
        from hindsight_api.engine.reflect.claude_agent import claude_reflect_agent
        assert callable(claude_reflect_agent)


def test_reflect_async_routes_to_existing_when_not_claude_code():
    """Verify non-claude-code providers still use run_reflect_agent."""
    from hindsight_api.engine.reflect.agent import run_reflect_agent
    assert callable(run_reflect_agent)
```

**Step 2: Run test to verify it fails / passes (routing sanity check)**

Run: `cd hindsight-api && python -m pytest tests/test_claude_sdk_integration.py -v --timeout 30 -n 0`

**Step 3: Modify `memory_engine.py` — reflect_async() integration**

At line ~4682 (where `run_reflect_agent()` is called), add conditional:

```python
# In reflect_async(), replace the direct run_reflect_agent() call:

if self._reflect_llm_config.provider == "claude-code":
    from .reflect.claude_agent import claude_reflect_agent
    agent_result = await claude_reflect_agent(
        query=query,
        bank_profile=profile,
        search_mental_models_fn=search_mental_models_fn,
        search_observations_fn=search_observations_fn,
        recall_fn=recall_fn,
        expand_fn=expand_fn,
        context=context,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        directives=directives,
        has_mental_models=has_mental_models,
        budget=effective_budget,
    )
else:
    agent_result = await run_reflect_agent(
        llm_config=self._reflect_llm_config,
        # ... existing args unchanged ...
    )
```

Similarly for retain (line ~1936) and consolidation (in consolidator.py line ~577).

> **Important:** The exact integration for retain and consolidation depends on how deeply they're called. Retain goes through `orchestrator.py` → `fact_extraction.py`. The conditional should be at the orchestrator level where `llm_config` is available. Verify exact insertion points during implementation.

**Step 4: Run existing tests to verify no regressions**

Run: `cd hindsight-api && python -m pytest tests/test_reflect_agent.py tests/test_retain.py tests/test_consolidation.py -v --timeout 300 -n 4`
Expected: All existing tests PASS (non-claude-code paths unchanged)

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/memory_engine.py tests/test_claude_sdk_integration.py
git commit -m "feat: wire claude-code provider conditional routing in memory_engine"
```

---

## Task 7: Streamline Reflect System Prompt

**Files:**
- Modify: `engine/reflect/prompts.py` (lines ~95-500)
- Test: `tests/test_claude_reflect_agent.py` (add prompt test)

**Step 1: Write the failing test**

```python
# Append to tests/test_claude_reflect_agent.py

from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools


def test_system_prompt_no_forced_workflow_steps():
    """Verify streamlined prompt doesn't contain forced numbered workflow."""
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
```

**Step 2: Run — should pass with existing code (baseline)**

Run: `cd hindsight-api && python -m pytest tests/test_claude_reflect_agent.py::test_system_prompt_no_forced_workflow_steps -v --timeout 30 -n 0`

**Step 3: Streamline the prompt**

In `prompts.py`, modify `build_system_prompt_for_tools()`:
- Keep: anti-hallucination, HIERARCHICAL RETRIEVAL STRATEGY, directives, bank profile, research depth
- Remove: forced numbered workflow steps that reference specific iteration ordering
- Reword: `"This is a NON-CONVERSATIONAL system"` → `"Do not ask clarifying questions. Gather evidence using the available tools and call done() with your answer."`
- Keep `FINAL_SYSTEM_PROMPT` and `build_final_prompt()` in file (still used by non-claude-code path) — do NOT delete

**Step 4: Run all reflect tests**

Run: `cd hindsight-api && python -m pytest tests/test_reflect_agent.py tests/test_claude_reflect_agent.py -v --timeout 300 -n 4`
Expected: PASS

**Step 5: Commit**

```bash
cd hindsight-api
git add hindsight_api/engine/reflect/prompts.py tests/test_claude_reflect_agent.py
git commit -m "refactor: streamline reflect system prompt for SDK auto-loop"
```

---

## Task 8: Full Regression Test

**Files:** None (test-only)

**Step 1: Run full test suite**

Run: `cd hindsight-api && python -m pytest tests/ -v --timeout 300 -n 8`
Expected: All tests PASS

**Step 2: Fix any failures**

Address import errors, missing adapters, type issues.

**Step 3: Commit fixes if any**

```bash
cd hindsight-api
git add -A
git commit -m "fix: address regression issues from Claude SDK migration"
```

---

## Execution Order Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Shared concurrency semaphore | — |
| 2 | Reflect tool builders + handlers | 1 |
| 3 | `claude_reflect_agent()` main function | 1, 2 |
| 4 | `claude_retain_agent()` | 1 |
| 5 | `claude_consolidate_agent()` | 1 |
| 6 | Integration wiring in `memory_engine.py` | 2, 3, 4, 5 |
| 7 | Streamline reflect system prompt | 3 |
| 8 | Full regression test | 6, 7 |

## Notes for Implementer

- **Import paths:** All `claude_agent_sdk` imports use underscores: `from claude_agent_sdk import ClaudeSDKClient`
- **ToolCall dataclass:** Verify exact import path — may be in `reflect/agent.py` or `reflect/types.py`
- **ReflectAgentResult:** Verify exact fields and constructor — adapt `result_holder` accordingly
- **`_build_user_message`:** Currently in `fact_extraction.py` — verify it's importable and compatible
- **Consolidation prompt split:** `build_batch_consolidation_prompt()` returns a template with `{facts_text}` and `{observations_text}` placeholders. Need to split into system prompt (rules) and user prompt (data). Verify during implementation.
- **Existing tests must not break:** The non-claude-code path is UNCHANGED. All existing tests should pass without modification.
