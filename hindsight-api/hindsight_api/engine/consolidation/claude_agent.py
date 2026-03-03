"""Claude Agent SDK consolidation agent with real consolidate tool handler.

This module provides `claude_consolidate_agent()` which replaces `_consolidate_batch_with_llm()`
when the provider is configured as `claude-code`. Instead of a one-shot JSON call, it uses
a long-lived ClaudeSDKClient session with a real MCP tool handler that Claude calls to submit
its consolidation decisions.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        from ..claude_sdk_utils import get_claude_sdk_semaphore

        _semaphore = get_claude_sdk_semaphore()
    return _semaphore


def build_consolidate_tool(*, result_holder: list[dict[str, Any]]) -> Any:
    """Build a consolidate MCP tool that captures Claude's consolidation decisions.

    The tool handler appends the submitted actions to `result_holder` so the caller
    can inspect the result after the agent run completes.

    Args:
        result_holder: A mutable list that will receive one dict with
            "creates", "updates", and "deletes" keys when Claude calls the tool.

    Returns:
        SdkMcpTool instance ready to be registered in an MCP server.
    """
    from claude_agent_sdk import SdkMcpTool

    async def handler(args: dict[str, Any]) -> str:
        creates = args.get("creates") or []
        updates = args.get("updates") or []
        deletes = args.get("deletes") or []
        result_holder.append(
            {
                "creates": creates,
                "updates": updates,
                "deletes": deletes,
            }
        )
        n_creates = len(creates)
        n_updates = len(updates)
        n_deletes = len(deletes)
        return (
            f"Consolidation accepted. "
            f"{n_creates} create(s), {n_updates} update(s), {n_deletes} delete(s) recorded."
        )

    return SdkMcpTool(
        name="consolidate",
        description=(
            "Submit consolidation results: create new observations, update existing ones, "
            "or delete obsolete ones. Call this exactly once with all decisions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "creates": {
                    "type": "array",
                    "description": "New observations to create.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Observation text."},
                            "source_fact_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "UUIDs of source facts from NEW FACTS list.",
                            },
                        },
                        "required": ["text", "source_fact_ids"],
                    },
                },
                "updates": {
                    "type": "array",
                    "description": "Existing observations to update.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {
                                "type": "string",
                                "description": "UUID of the existing observation to update.",
                            },
                            "text": {"type": "string", "description": "New observation text."},
                            "source_fact_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "UUIDs of source facts from NEW FACTS list.",
                            },
                        },
                        "required": ["observation_id", "text", "source_fact_ids"],
                    },
                },
                "deletes": {
                    "type": "array",
                    "description": "Observations to delete (superseded or contradicted).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {
                                "type": "string",
                                "description": "UUID of the observation to remove.",
                            },
                        },
                        "required": ["observation_id"],
                    },
                },
            },
            "required": ["creates", "updates", "deletes"],
        },
        handler=handler,
    )


def _build_system_prompt(observations_mission: str | None) -> str:
    """Build the system prompt (rules portion) for the consolidation agent.

    Extracts the rules section from the full consolidation prompt template so
    Claude receives stable instructions as a system prompt, separate from
    the per-call data (facts + observations).
    """
    from .prompts import _DEFAULT_MISSION, _PROCESSING_RULES

    mission = observations_mission or _DEFAULT_MISSION
    return (
        "You are a memory consolidation system. Synthesize facts into observations "
        "and merge with existing observations when appropriate.\n\n"
        f"## MISSION\n{mission}\n\n"
        f"{_PROCESSING_RULES}\n\n"
        "When you have analyzed the facts and existing observations, call the `consolidate` tool "
        "exactly once with your decisions (creates/updates/deletes). "
        "If nothing durable is found, call it with empty arrays."
    )


def _build_user_message(memories: list[dict[str, Any]], observations_text: str) -> str:
    """Build the per-call user message with facts and existing observations."""

    def _fact_line(m: dict[str, Any]) -> str:
        parts = [f"[{m.get('id', '')}] {m.get('text', '')}"]
        if m.get("occurred_start"):
            parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            parts.append(f"mentioned_at={m['mentioned_at']}")
        return " | ".join(parts)

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    return (
        f"NEW FACTS:\n{facts_lines}\n\n"
        f"EXISTING OBSERVATIONS (JSON array, pooled from recalls across all facts above):\n"
        f"{observations_text}\n\n"
        "Each observation includes:\n"
        "- id: unique identifier for updating\n"
        "- text: the observation content\n"
        "- proof_count: number of supporting memories\n"
        "- occurred_start/occurred_end: temporal range of source facts\n"
        "- source_memories: array of supporting facts with their text and dates\n\n"
        "Compare the facts against existing observations:\n"
        "- Same topic as an existing observation → UPDATE it (observation_id + source_fact_ids)\n"
        "- New topic with durable knowledge → CREATE a new observation (source_fact_ids)\n"
        "- Cross-reference facts within the batch: a later fact may resolve a vague reference in an earlier one\n"
        "- Purely ephemeral facts → omit them (no create/update needed)\n\n"
        "Now call the `consolidate` tool with your decisions."
    )


async def claude_consolidate_agent(
    memories: list[dict[str, Any]],
    observations_text: str,
    observations_mission: str | None = None,
) -> dict[str, Any]:
    """Run batch consolidation using Claude Agent SDK with a real MCP tool handler.

    This is the SDK-native replacement for `_consolidate_batch_with_llm()` when
    the provider is `claude-code`. Claude receives:
    - System prompt: stable consolidation rules + mission
    - User message: current batch of facts + retrieved observations
    - MCP tool: `consolidate` to submit decisions

    Args:
        memories: List of memory dicts with at least "id" and "text" keys.
        observations_text: JSON-serialized list of existing observations.
        observations_mission: Optional bank-specific mission override.

    Returns:
        Dict with "creates", "updates", "deletes" lists (empty if no durable knowledge found).
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        create_sdk_mcp_server,
    )

    result_holder: list[dict[str, Any]] = []
    tool = build_consolidate_tool(result_holder=result_holder)
    mcp_server = create_sdk_mcp_server(
        name="hindsight_consolidation",
        version="1.0.0",
        tools=[tool],
    )

    system_prompt = _build_system_prompt(observations_mission)
    user_message = _build_user_message(memories, observations_text)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=3,
        mcp_servers={"hindsight_consolidation": mcp_server},
        allowed_tools=["mcp__hindsight_consolidation__consolidate"],
    )

    async with _get_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_message)
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break

    if result_holder:
        return result_holder[0]

    logger.warning("[CONSOLIDATION] Claude did not call consolidate tool — returning empty result")
    return {"creates": [], "updates": [], "deletes": []}
