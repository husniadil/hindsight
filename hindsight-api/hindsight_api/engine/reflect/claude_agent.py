"""Claude Agent SDK reflect agent with real MCP tool handlers.

Task 2: build_reflect_tools() — real tool handlers as closures
Task 3: claude_reflect_agent() — main function with SDK auto-loop
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    TextBlock,
    create_sdk_mcp_server,
)

from ..claude_sdk_utils import get_claude_sdk_semaphore
from .models import ReflectAgentResult, ToolCall
from .prompts import build_system_prompt_for_tools

logger = logging.getLogger(__name__)

_semaphore_instance: Any = None


def _get_semaphore() -> Any:
    global _semaphore_instance
    if _semaphore_instance is None:
        _semaphore_instance = get_claude_sdk_semaphore()
    return _semaphore_instance


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
        reason = str(args.get("reason", ""))
        t0 = time.time()
        try:
            data = await search_mental_models_fn(query, max_results)  # type: ignore[misc]
            ids = _extract_ids_from_results(data, "mental_models")
            available_mental_model_ids.update(ids)
            tool_trace.append(ToolCall(
                tool="search_mental_models",
                reason=reason,
                input={"query": query, "max_results": max_results},
                output=data,
                duration_ms=int((time.time() - t0) * 1000),
                iteration=len(tool_trace),
            ))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"search_mental_models error: {exc}")
            return json.dumps({"error": str(exc)})

    async def search_observations_handler(args: dict) -> str:
        query = str(args.get("query", ""))
        max_tokens = int(args.get("max_tokens", 5000))
        reason = str(args.get("reason", ""))
        t0 = time.time()
        try:
            data = await search_observations_fn(query, max_tokens)  # type: ignore[misc]
            ids = _extract_ids_from_results(data, "observations")
            available_observation_ids.update(ids)
            tool_trace.append(ToolCall(
                tool="search_observations",
                reason=reason,
                input={"query": query, "max_tokens": max_tokens},
                output=data,
                duration_ms=int((time.time() - t0) * 1000),
                iteration=len(tool_trace),
            ))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"search_observations error: {exc}")
            return json.dumps({"error": str(exc)})

    async def recall_handler(args: dict) -> str:
        query = str(args.get("query", ""))
        max_tokens = int(args.get("max_tokens", 2048))
        max_chunk_tokens = int(args.get("max_chunk_tokens", 1000))
        reason = str(args.get("reason", ""))
        t0 = time.time()
        try:
            data = await recall_fn(query, max_tokens, max_chunk_tokens)  # type: ignore[misc]
            ids = _extract_ids_from_results(data, "memories")
            available_memory_ids.update(ids)
            tool_trace.append(ToolCall(
                tool="recall",
                reason=reason,
                input={"query": query, "max_tokens": max_tokens, "max_chunk_tokens": max_chunk_tokens},
                output=data,
                duration_ms=int((time.time() - t0) * 1000),
                iteration=len(tool_trace),
            ))
            return json.dumps(data, default=str)
        except Exception as exc:
            logger.warning(f"recall error: {exc}")
            return json.dumps({"error": str(exc)})

    async def expand_handler(args: dict) -> str:
        memory_ids = args.get("memory_ids", [])
        depth = str(args.get("depth", "chunk"))
        reason = str(args.get("reason", ""))
        t0 = time.time()
        try:
            data = await expand_fn(memory_ids, depth)  # type: ignore[misc]
            tool_trace.append(ToolCall(
                tool="expand",
                reason=reason,
                input={"memory_ids": memory_ids, "depth": depth},
                output=data,
                duration_ms=int((time.time() - t0) * 1000),
                iteration=len(tool_trace),
            ))
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
            text=answer,
            used_memory_ids=list(cited_memory_ids & available_memory_ids),
            used_mental_model_ids=list(cited_mental_model_ids & available_mental_model_ids),
            used_observation_ids=list(cited_observation_ids & available_observation_ids),
            tool_trace=list(tool_trace),
            tools_called=len(tool_trace),
        )
        result_holder.append(result)
        return "Answer accepted."

    # Import tool schemas from existing module and build SdkMcpTool list
    from ..reflect.tools_schema import get_reflect_tools

    schemas = get_reflect_tools()  # returns list of OpenAI-format tool dicts

    handler_map: dict[str, Any] = {
        "search_mental_models": search_mental_models_handler,
        "search_observations": search_observations_handler,
        "recall": recall_handler,
        "expand": expand_handler,
        "done": done_handler,
    }

    tools = []
    for schema in schemas:
        func = schema.get("function", {})
        name = func.get("name", "")
        if name in handler_map:
            tools.append(SdkMcpTool(
                name=name,
                description=func.get("description", ""),
                input_schema=func.get("parameters", {}),
                handler=handler_map[name],
            ))
    return tools


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

    # Build system prompt (reuse existing)
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
        text=fallback_text.strip() if fallback_text else "I could not find relevant information.",
        used_memory_ids=list(available_memory_ids),
        used_mental_model_ids=list(available_mental_model_ids),
        used_observation_ids=list(available_observation_ids),
        tool_trace=tool_trace,
        tools_called=len(tool_trace),
    )
