# Hindsight: Claude Agent SDK Long-lived Session Migration

**Date:** 2026-03-03
**Status:** Approved
**Scope:** Reflect, Retain, Consolidation — Claude-code provider path only

## Summary

Migrate Hindsight's Claude Agent SDK usage from stateless one-shot sessions (10+ sessions per reflect, manual agentic loop, dummy tool handlers) to long-lived sessions with real MCP tool handlers and SDK-managed multi-turn conversations. Non-claude-code providers remain unchanged.

## Goals

1. Eliminate manual agentic loop in reflect agent (~200 lines → ~30 lines)
2. Real MCP tool handlers — SDK executes tools directly, no Python dispatch loop
3. Sequential chunk processing for retain with cross-chunk context continuity
4. Tool-based structured output (SDK-enforced schema) replacing prompt-hack JSON
5. Preserve multi-provider support — conditional path, zero interface changes

## Architecture

### Dual-path Routing

```
provider == "claude-code"?
├── YES → claude_*_agent() modules (NEW)
│         ├── 1 ClaudeSDKClient per method call
│         ├── Real MCP tools as closures
│         ├── SDK auto-loops, auto-compacts
│         ├── Bypasses LLMProvider + global semaphore
│         └── Dedicated _claude_sdk_semaphore
│
└── NO → existing LLMProvider path (UNCHANGED)
```

### Why Not Singleton

A true singleton ClaudeSDKClient is wrong for Hindsight because:

1. **Per-request tool state**: Tool handlers need `bank_id`, `tags`, `request_context` — all differ per request. MCP servers are fixed at client creation.
2. **Concurrency**: Concurrent requests would clobber shared mutable state.
3. **Context leaking**: Session 1 context would bleed into session 2.
4. **Semaphore blocking**: SDK auto-loop holding global semaphore for 30-60s blocks all other LLM calls.

Per-method-call client is correct. "Long-lived" benefit is **within** a reflect/retain call (multi-turn context persistence), not across calls.

## Reflect Agent — `claude_reflect_agent()`

### File: `engine/reflect/claude_agent.py`

**Signature:**

```python
async def claude_reflect_agent(
    query: str,
    bank_profile: dict[str, Any],
    search_mental_models_fn: Callable,
    search_observations_fn: Callable,
    recall_fn: Callable,
    expand_fn: Callable,
    context: str | None = None,
    max_iterations: int = 10,
    max_tokens: int | None = None,
    directives: list[dict] | None = None,
    has_mental_models: bool = False,
    budget: str | None = None,
) -> ReflectAgentResult:
```

### Closure-captured Shared State

```python
available_memory_ids: set[str] = set()
available_mental_model_ids: set[str] = set()
available_observation_ids: set[str] = set()
tool_trace: list[ToolCall] = []
result_holder: list[ReflectAgentResult] = []
```

### 5 Real MCP Tool Handlers

| Tool | Handler |
|------|---------|
| `search_mental_models` | Calls `search_mental_models_fn(query, max_results)` → populates `available_mental_model_ids` → appends to `tool_trace` → returns JSON |
| `search_observations` | Calls `search_observations_fn(query, max_tokens)` → populates `available_observation_ids` → returns JSON |
| `recall` | Calls `recall_fn(query, max_tokens, max_chunk_tokens)` → populates `available_memory_ids` → returns JSON |
| `expand` | Calls `expand_fn(memory_ids, depth)` → returns JSON |
| `done` | Validates cited IDs against `available_*` sets → builds `ReflectAgentResult` → stores in `result_holder` → returns `"Answer accepted."` |

### SDK Session

```python
async with _claude_sdk_semaphore:
    mcp_server = create_sdk_mcp_server(name="hindsight", tools=sdk_tools)
    options = ClaudeAgentOptions(
        system_prompt=build_system_prompt_for_tools(...),
        max_turns=max_iterations + 2,
        mcp_servers={"hindsight": mcp_server},
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(query)
        async for msg in client.receive_messages():
            if isinstance(msg, ResultMessage):
                break

    if result_holder:
        return result_holder[0]
    # fallback: max_turns hit without done()
```

### System Prompt Changes

- **Keep**: Anti-hallucination, HIERARCHICAL RETRIEVAL STRATEGY, directives, bank profile, research depth budget
- **Remove**: Forced workflow numbered steps (1-5), `FINAL_SYSTEM_PROMPT`, `build_final_prompt()`
- **Reword**: `"NON-CONVERSATIONAL"` → `"Do not ask clarifying questions. Gather evidence and call done()."`
- **Remove**: `_is_context_overflow_error()` — SDK auto-compact handles this

### Eliminated Code

- `FINAL_SYSTEM_PROMPT` constant
- `build_final_prompt()` function
- `build_agent_prompt()` function (legacy/unused)
- `_is_context_overflow_error()` helper
- Forced `tool_choice` per-iteration logic
- Manual message list management

## Retain Agent — `claude_retain_agent()`

### File: `engine/retain/claude_agent.py`

**Signature:**

```python
async def claude_retain_agent(
    text: str,
    event_date: datetime | None,
    context: str,
    config: HindsightConfig,
    metadata: dict[str, str] | None = None,
) -> tuple[list[Fact], list[ChunkMetadata], TokenUsage]:
```

### Flow

1. `chunk_text(text, max_chars=config.retain_chunk_size)` — preprocessing preserved
2. 1 `ClaudeSDKClient` per retain call
3. Sequential `query()` per chunk in same session
4. Claude sees previous extractions in context → consistent entity naming, dedup

### extract_facts Tool

Permissive `input_schema` (only `what` required, `additionalProperties: true`). Handler does lenient parsing:

- Fallback field names (`what` → `factual_core`)
- Auto-fix `fact_type` to valid values
- Entity normalization
- Skip malformed facts (don't reject entire call)

### Sequential Chunk Processing

```python
async with _claude_sdk_semaphore:
    async with ClaudeSDKClient(options=options) as client:
        for i, chunk in enumerate(chunks):
            user_msg = build_chunk_prompt(chunk, i, len(chunks), event_date, context, metadata)
            await client.query(user_msg)
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break
```

### Trade-off: Sequential vs Parallel

- Sequential is ~5x slower than current parallel `asyncio.gather`
- Acceptable: retain is background ingestion, not latency-critical
- Benefit: Cross-chunk context continuity (entity resolution, consistent style)
- Parallelism preserved at document level (separate clients per document in batch)

## Consolidation Agent — `claude_consolidate_agent()`

### File: `engine/consolidation/claude_agent.py`

**Signature:**

```python
async def claude_consolidate_agent(
    memories: list[dict[str, Any]],
    union_observations: list[MemoryFact],
    union_source_facts: dict[str, MemoryFact],
    mission: str,
) -> _BatchLLMResult:
```

### Flow

1. 1 `ClaudeSDKClient` per consolidation batch
2. System prompt = consolidation rules (split from current user-only message)
3. User prompt = formatted facts + existing observations data
4. Claude calls `consolidate(creates=[], updates=[], deletes=[])` tool
5. Handler stores result, returns confirmation

### consolidate Tool

Structured `input_schema` matching `_ConsolidationBatchResponse`:
- `creates`: `[{text, source_fact_ids}]`
- `updates`: `[{observation_id, text, source_fact_ids}]`
- `deletes`: `[{observation_id}]`

## Concurrency Control

```python
_claude_sdk_semaphore = asyncio.Semaphore(
    int(os.getenv("HINDSIGHT_API_CLAUDE_SDK_MAX_CONCURRENT", "3"))
)
```

Separate from `_global_llm_semaphore`. All `claude_*_agent()` functions acquire this semaphore.

## Error Handling

| Scenario | Strategy |
|----------|----------|
| Auth error | Immediate raise (detect `"auth"`, `"login"`, `"credential"`) |
| LLM transient error | SDK handles retries internally |
| Context overflow | SDK auto-compact — no manual handling |
| Tool handler error | try/except → return `{"error": "..."}` → Claude sees error, can retry |
| `done()` without evidence | PreToolUse hook blocks if `available_*_ids` all empty |
| Max turns exceeded | SDK stops → fallback to last AssistantMessage text |

## Integration Points

### `memory_engine.py` — Conditional Branching

```python
# reflect_async()
if self._reflect_llm_config.provider == "claude-code":
    from .reflect.claude_agent import claude_reflect_agent
    agent_result = await claude_reflect_agent(
        query=query, bank_profile=profile,
        search_mental_models_fn=search_mental_models_fn,
        search_observations_fn=search_observations_fn,
        recall_fn=recall_fn, expand_fn=expand_fn,
        context=context, max_iterations=max_iterations,
        directives=directives, has_mental_models=has_mental_models,
        budget=effective_budget,
    )
else:
    agent_result = await run_reflect_agent(...)  # unchanged
```

Same pattern for retain and consolidation.

## File Changes

### New Files

| File | Est. Lines | Purpose |
|------|-----------|---------|
| `engine/reflect/claude_agent.py` | ~120 | `claude_reflect_agent()` + 5 real tool builders |
| `engine/retain/claude_agent.py` | ~100 | `claude_retain_agent()` + extract_facts tool |
| `engine/consolidation/claude_agent.py` | ~80 | `claude_consolidate_agent()` + consolidate tool |

### Modified Files

| File | Change |
|------|--------|
| `engine/memory_engine.py` | Conditional import + branch for claude-code provider |
| `engine/reflect/prompts.py` | Streamline system prompt (remove forced workflow, reword) |

### Unchanged Files

- `engine/reflect/agent.py` — fully preserved for non-claude-code providers
- `engine/reflect/tools_schema.py` — still used by existing path
- `engine/providers/claude_code_llm.py` — still available for any LLMProvider usage
- `engine/llm_wrapper.py` — LLMProvider interface unchanged
- All existing tests

## Gaps Analyzed and Resolved

| # | Gap | Resolution |
|---|-----|-----------|
| Singleton vs per-request | Per-request — tool state, concurrency, isolation |
| ID tracking | Closure-captured mutable sets, validated in done() handler |
| Tool/LLM trace | Handlers append to shared trace list |
| Runaway prevention | `max_turns` param + done-without-evidence hook |
| Context overflow | SDK auto-compact — eliminated as concern |
| Semaphore blocking | Dedicated `_claude_sdk_semaphore` |
| Lenient fact parsing | Permissive input_schema + lenient handler parsing |
| done() termination | Closure result_holder + confirmation string |
| Token usage granularity | SDK aggregate (more accurate than len//4 estimate) |
| Concurrent sessions | `_claude_sdk_semaphore` with configurable limit |
| Retain auto-split | Keep chunk_text(), auto-split deferred |
| Structured output reliability | Tool input_schema > prompt-hack — improvement |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Claude ignores retrieval strategy | Low | Medium | System prompt strategy + PreToolUse hook counter |
| Complex tool input_schema issues | Low | Low | Permissive schema, test iteratively |
| Sequential retain slower | Medium | Low | Background job, document-level parallelism |
| SDK behavior surprises | Medium | Medium | Build incrementally: reflect first, then retain, then consolidation |
