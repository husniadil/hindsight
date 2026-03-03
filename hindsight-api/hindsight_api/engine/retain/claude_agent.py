"""Claude Agent SDK retain agent with real extract_facts tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
)

from ..claude_sdk_utils import get_claude_sdk_semaphore
from .fact_extraction import Fact, chunk_text, _build_user_message
from .types import ChunkMetadata

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = get_claude_sdk_semaphore()
    return _semaphore


def build_extract_facts_tool(
    *,
    all_facts: list[Fact],
    chunk_metadata: list[ChunkMetadata],
) -> SdkMcpTool:
    """Build extract_facts tool with lenient handler.

    Handler does permissive parsing:
    - Accepts "what" or "factual_core" as the fact text field
    - Auto-fixes invalid fact_type to "world"
    - Skips malformed entries (no "what" field)
    """

    async def handler(args: dict[str, Any]) -> str:
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
        chunk_metadata.append(
            ChunkMetadata(
                chunk_text="",
                fact_count=len(parsed),
                content_index=0,
                chunk_index=len(chunk_metadata),
            )
        )
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
                            "fact_type": {
                                "type": "string",
                                "enum": ["world", "experience", "opinion"],
                            },
                        },
                        "required": ["what"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["facts"],
            "additionalProperties": True,
        },
        handler=handler,
    )


async def claude_retain_agent(
    text: str,
    event_date: datetime | None,
    context: str,
    config: Any,
    metadata: dict[str, str] | None = None,
) -> tuple[list[Fact], list[ChunkMetadata], dict[str, Any]]:
    """Run retain using Claude SDK with sequential chunks in same session."""
    chunks = chunk_text(text, max_chars=config.retain_chunk_size)
    all_facts: list[Fact] = []
    chunk_metadata: list[ChunkMetadata] = []

    tool = build_extract_facts_tool(all_facts=all_facts, chunk_metadata=chunk_metadata)
    mcp_server = create_sdk_mcp_server(name="hindsight_retain", version="1.0.0", tools=[tool])

    system_prompt = (
        getattr(config, "retain_system_prompt", None)
        or (
            "You are a fact extraction system. Extract significant facts from text chunks. "
            "Call extract_facts with the extracted facts for each chunk."
        )
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
