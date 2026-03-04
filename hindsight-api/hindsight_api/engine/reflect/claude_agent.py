"""Claude Agent SDK reflect agent with real MCP tool handlers.

Task 2: build_reflect_tools() — real tool handlers as closures
Task 3: claude_reflect_agent() — main function with SDK auto-loop
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..claude_sdk_utils import get_claude_sdk_semaphore, log_sdk_messages

if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider
from .models import ReflectAgentResult, ToolCall
from .prompts import build_system_prompt_for_tools

logger = logging.getLogger(__name__)


def _text_result(text: str) -> dict[str, Any]:
    """Return SDK-compatible tool result dict."""
    return {"content": [{"type": "text", "text": text}]}


def _extract_ids_from_results(data: dict, key: str, id_field: str = "id") -> set[str]:
    """Extract IDs from tool result dict."""
    ids: set[str] = set()
    items = data.get(key, [])
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and id_field in item:
                ids.add(str(item[id_field]))
    return ids


_BUDGET_EXHAUSTED_MSG = (
    "Budget exhausted ({used}/{limit} tool calls used). "
    "You MUST call done() now with your answer based on evidence gathered so far."
)


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
    max_search_calls: int | None = None,
) -> list[Any]:
    """Build real MCP tool handlers as closures capturing shared mutable state.

    Args:
        max_search_calls: Maximum number of search/recall/expand tool calls allowed.
            When reached, tools return a budget-exhausted message forcing done().
            None means unlimited.
    """
    from claude_agent_sdk import SdkMcpTool

    def _check_budget() -> dict[str, Any] | None:
        """Return budget-exhausted result if limit reached, else None."""
        if max_search_calls is not None and len(tool_trace) >= max_search_calls:
            return _text_result(json.dumps({
                "error": _BUDGET_EXHAUSTED_MSG.format(
                    used=len(tool_trace), limit=max_search_calls,
                ),
            }))
        return None

    async def search_mental_models_handler(args: dict) -> dict[str, Any]:
        if (gate := _check_budget()) is not None:
            return gate
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
            return _text_result(json.dumps(data, default=str))
        except Exception as exc:
            logger.warning(f"search_mental_models error: {exc}")
            return _text_result(json.dumps({"error": str(exc)}))

    async def search_observations_handler(args: dict) -> dict[str, Any]:
        if (gate := _check_budget()) is not None:
            return gate
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
            return _text_result(json.dumps(data, default=str))
        except Exception as exc:
            logger.warning(f"search_observations error: {exc}")
            return _text_result(json.dumps({"error": str(exc)}))

    async def recall_handler(args: dict) -> dict[str, Any]:
        if (gate := _check_budget()) is not None:
            return gate
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
            return _text_result(json.dumps(data, default=str))
        except Exception as exc:
            logger.warning(f"recall error: {exc}")
            return _text_result(json.dumps({"error": str(exc)}))

    async def expand_handler(args: dict) -> dict[str, Any]:
        if (gate := _check_budget()) is not None:
            return gate
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
            return _text_result(json.dumps(data, default=str))
        except Exception as exc:
            logger.warning(f"expand error: {exc}")
            return _text_result(json.dumps({"error": str(exc)}))

    async def done_handler(args: dict) -> dict[str, Any]:
        # Block if no evidence gathered
        if not available_memory_ids and not available_mental_model_ids and not available_observation_ids:
            return _text_result(json.dumps({
                "error": "You must search for information first. "
                "Use search_mental_models(), search_observations(), or recall() before providing your final answer."
            }))

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
            iterations=len(tool_trace),
        )
        result_holder.append(result)
        return _text_result("Answer accepted.")

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
    max_tokens: int | None = None,  # accepted for API parity; not used by SDK path
    response_schema: dict | None = None,
    llm_config: LLMProvider | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    budget: str | None = None,
) -> ReflectAgentResult:
    """Run reflect agent using Claude SDK with real MCP tool handlers.

    Creates a ClaudeSDKClient per call. SDK auto-loops tool calls until done() is called.

    Budget enforcement uses two layers:
    - Layer 1 (tool-level gate): search/recall/expand tools return a budget-exhausted
      message after ``max_search_calls`` calls, forcing the agent to call done().
    - Layer 2 (max_turns safety net): SDK hard-stops after ``max_turns`` to prevent
      infinite loops if the agent ignores the gate.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        TextBlock,
        create_sdk_mcp_server,
    )

    # Budget: reserve 1 iteration for synthesis (done call), rest for search tools.
    # max_turns adds buffer for done() call + potential retries.
    max_search_calls = max(1, max_iterations - 1)
    max_turns = max_iterations + 2

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
        max_search_calls=max_search_calls,
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
        mcp_servers={"hindsight": mcp_server},
        allowed_tools=[f"mcp__hindsight__{t.name}" for t in sdk_tools],
        max_turns=max_turns,
    )

    fallback_text = ""

    async with get_claude_sdk_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            await client.query(query)
            async for msg in client.receive_response():
                log_sdk_messages(msg, agent_name="reflect", log=logger)
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            fallback_text = block.text

    # Return result from done() handler, or fallback
    if result_holder:
        result = result_holder[0]
    else:
        # Fallback: max_turns hit without done() — use last assistant text
        logger.warning("Claude reflect agent completed without calling done()")
        result = ReflectAgentResult(
            text=fallback_text.strip() if fallback_text else "I could not find relevant information.",
            used_memory_ids=list(available_memory_ids),
            used_mental_model_ids=list(available_mental_model_ids),
            used_observation_ids=list(available_observation_ids),
            tool_trace=tool_trace,
            tools_called=len(tool_trace),
            iterations=len(tool_trace),
        )

    # Generate structured output if response_schema provided (matches non-SDK path behavior)
    if response_schema and llm_config and result.text:
        try:
            from .agent import _generate_structured_output

            structured_output, _, _ = await _generate_structured_output(
                result.text, response_schema, llm_config, "claude-sdk"
            )
            result.structured_output = structured_output
        except Exception as exc:
            logger.warning(f"Claude reflect agent structured output failed: {exc}")

    return result
