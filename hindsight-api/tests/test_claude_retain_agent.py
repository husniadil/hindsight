"""Tests for claude_retain_agent — build_extract_facts_tool and claude_retain_agent."""
from __future__ import annotations

import pytest
from hindsight_api.engine.retain.claude_agent import build_extract_facts_tool


@pytest.mark.asyncio
async def test_extract_facts_handler_lenient_parsing():
    from hindsight_api.engine.retain.fact_extraction import Fact
    from hindsight_api.engine.retain.types import ChunkMetadata

    all_facts: list[Fact] = []
    chunk_metadata: list[ChunkMetadata] = []

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
    from hindsight_api.engine.retain.fact_extraction import Fact
    from hindsight_api.engine.retain.types import ChunkMetadata

    all_facts: list[Fact] = []
    chunk_metadata: list[ChunkMetadata] = []

    tool = build_extract_facts_tool(all_facts=all_facts, chunk_metadata=chunk_metadata)
    result = await tool.handler({"facts": []})
    assert len(all_facts) == 0
    assert "0 facts" in result.lower() or "extracted 0" in result.lower()
