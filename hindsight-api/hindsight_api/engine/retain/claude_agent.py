"""Claude Agent SDK retain agent with real extract_facts tool handler."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..claude_sdk_utils import get_claude_sdk_semaphore, log_sdk_messages
from .fact_extraction import Entity, Fact, chunk_text, _build_user_message
from .types import ChunkMetadata

logger = logging.getLogger(__name__)


def build_extract_facts_tool(
    *,
    all_facts: list[Fact],
    chunk_metadata: list[ChunkMetadata],
    current_chunk: list[str],
) -> Any:
    """Build extract_facts tool with lenient handler.

    Handler does permissive parsing:
    - Accepts "what" or "factual_core" as the fact text field
    - Auto-fixes invalid fact_type to "world"
    - Parses optional fields: occurred_start/end, where, entities
    - Skips malformed entries (no "what" field)
    """
    from claude_agent_sdk import SdkMcpTool

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
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

            # Parse optional fields
            entities = None
            raw_entities = fact_data.get("entities")
            if isinstance(raw_entities, list):
                entities = [Entity(text=str(e)) for e in raw_entities if e]

            try:
                # causal_relations intentionally omitted — requires cross-fact index
                # references that are unreliable in tool-call output. The integration
                # layer defaults to [] which is safe.
                fact = Fact(
                    fact=str(what),
                    fact_type=fact_type,
                    occurred_start=fact_data.get("occurred_start"),
                    occurred_end=fact_data.get("occurred_end"),
                    where=fact_data.get("where"),
                    entities=entities,
                )
                parsed.append(fact)
            except Exception:
                continue

        all_facts.extend(parsed)
        chunk_metadata.append(
            ChunkMetadata(
                chunk_text=current_chunk[0] if current_chunk else "",
                fact_count=len(parsed),
                content_index=0,
                chunk_index=len(chunk_metadata),
            )
        )
        return {"content": [{"type": "text", "text": f"Extracted {len(parsed)} facts."}]}

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
                            "what": {
                                "type": "string",
                                "description": "Core fact - concise but complete (1-2 sentences). "
                                "Include who, when, where, why details inline.",
                            },
                            "fact_type": {
                                "type": "string",
                                "enum": ["world", "experience", "opinion"],
                                "description": "'world' = objective facts, 'experience' = personal experiences, "
                                "'opinion' = subjective views",
                            },
                            "occurred_start": {
                                "type": ["string", "null"],
                                "description": "ISO 8601 timestamp when the event started, if known. null otherwise.",
                            },
                            "occurred_end": {
                                "type": ["string", "null"],
                                "description": "ISO 8601 timestamp when the event ended, if known. null otherwise.",
                            },
                            "where": {
                                "type": ["string", "null"],
                                "description": "Location where the fact occurred, if relevant. null otherwise.",
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Named entities: people, places, organizations, concepts mentioned.",
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
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )

    chunks = chunk_text(text, max_chars=config.retain_chunk_size)
    all_facts: list[Fact] = []
    chunk_metadata: list[ChunkMetadata] = []
    current_chunk: list[str] = [""]

    tool = build_extract_facts_tool(all_facts=all_facts, chunk_metadata=chunk_metadata, current_chunk=current_chunk)
    mcp_server = create_sdk_mcp_server(name="hindsight_retain", version="1.0.0", tools=[tool])

    system_prompt = (
        getattr(config, "retain_system_prompt", None)
        or (
            "You are a fact extraction system. Extract significant facts from text chunks. "
            "Call extract_facts with the extracted facts for each chunk. "
            "For each fact, extract: the core statement (what), fact_type, temporal info "
            "(occurred_start/end as ISO timestamps), location (where), and named entities."
        )
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"hindsight_retain": mcp_server},
        allowed_tools=["mcp__hindsight_retain__extract_facts"],
    )

    async with get_claude_sdk_semaphore():
        async with ClaudeSDKClient(options=options) as client:
            for i, chunk in enumerate(chunks):
                current_chunk[0] = chunk
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
                    log_sdk_messages(msg, agent_name="retain", log=logger)

    return all_facts, chunk_metadata, {}
