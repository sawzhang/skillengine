# Roadmap

SkillEngine improvement roadmap, based on architectural comparison with [pi-mono](https://github.com/badlogic/pi-mono) (a production-grade multi-provider agent SDK in TypeScript).

Our core advantage — Markdown+YAML skill definition, eligibility filtering, deterministic actions, hot-reload — remains the differentiator. The improvements below focus on **agent runtime capabilities** to reach production-grade extensibility.

---

## P0 — Event System

**Problem**: The agent loop is a black box. Callers cannot intercept tool execution, modify context, or observe lifecycle transitions. Without events, any extension/plugin system is fundamentally limited.

**Current state**: No event mechanism. `AgentRunner.chat()` runs to completion with no hooks.

**Target design**:

```python
class EventBus:
    def on(self, event: str, handler: Callable) -> Callable  # returns unsubscribe
    def emit(self, event: str, data: Any) -> EventResult

class AgentRunner:
    events: EventBus

# Handlers can return control signals
@agent.events.on("before_tool_call")
async def guard(event: ToolCallEvent) -> ToolCallEventResult:
    if event.tool_name == "bash" and "rm -rf" in event.args["command"]:
        return ToolCallEventResult(block=True, reason="Dangerous command blocked")
    return ToolCallEventResult()
```

**Events to implement** (minimum viable set):

| Event | Timing | Handler can... |
|-------|--------|----------------|
| `agent_start` | Before first LLM call | Inspect config |
| `agent_end` | After agent finishes | Access final state |
| `turn_start` | Before each LLM round | Modify messages |
| `turn_end` | After each LLM round | Inspect response |
| `before_tool_call` | Before tool execution | Block, modify args |
| `after_tool_result` | After tool returns | Modify result |
| `context_transform` | Before sending to LLM | Prune/inject messages |
| `input` | User message received | Transform or intercept |

**Files to create/modify**:
- `src/skillengine/events.py` — `EventBus`, event types, result types
- `src/skillengine/agent.py` — Integrate events into `AgentRunner` loop
- `tests/test_events.py` — Event emission order, handler return values

---

## P0 — Structured Stream Events

**Problem**: `chat_stream()` yields `AsyncIterator[str]` — plain text deltas only. Cannot distinguish thinking, text, tool calls, or errors. UI cannot render them differently.

**Current state**:

```python
async for chunk in agent.chat_stream("hello"):
    print(chunk, end="")  # All chunks are opaque strings
```

**Target design**:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class StreamEvent:
    type: Literal[
        "text_start", "text_delta", "text_end",
        "thinking_start", "thinking_delta", "thinking_end",
        "tool_call_start", "tool_call_delta", "tool_call_end",
        "tool_result",
        "done", "error",
    ]
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    error: Exception | None = None

# Usage
async for event in agent.chat_stream_events("hello"):
    match event.type:
        case "text_delta":
            print(event.content, end="")
        case "thinking_delta":
            print(f"[thinking] {event.content}", end="")
        case "tool_call_start":
            print(f"\n> Calling {event.tool_name}...")
        case "error":
            print(f"Error: {event.error}")
```

**Implementation notes**:
- Keep existing `chat_stream()` as a convenience wrapper (yields only text deltas)
- Add `chat_stream_events()` returning `AsyncIterator[StreamEvent]`
- Each `LLMAdapter` must map provider-specific events to `StreamEvent`

**Files to create/modify**:
- `src/skillengine/models.py` — Add `StreamEvent` dataclass
- `src/skillengine/adapters/base.py` — Add `chat_stream_events()` to interface
- `src/skillengine/adapters/openai.py` — Map OpenAI stream chunks
- `src/skillengine/adapters/anthropic.py` — Map Anthropic stream events

---

## P1 — Model Metadata & Registry

**Problem**: Models are just a string name + base_url. No metadata (cost, context window, capabilities). Cannot make intelligent decisions about model selection, cost tracking, or context overflow.

**Current state**:

```python
config = AgentConfig(model="MiniMax-M2.1", base_url="...", api_key="...")
```

**Target design**:

```python
@dataclass
class ModelDefinition:
    id: str                        # "gpt-4o", "claude-sonnet-4-20250514"
    provider: str                  # "openai", "anthropic", "minimax"
    api: str                       # "openai-compatible", "anthropic-messages"
    display_name: str              # Human-readable
    context_window: int            # Max input tokens
    max_output_tokens: int
    cost: ModelCost                # $/million tokens (input, output, cache)
    capabilities: set[str]         # {"text", "image", "reasoning", "tool_use"}
    reasoning: bool = False        # Supports extended thinking

@dataclass
class ModelCost:
    input: float                   # $/million input tokens
    output: float                  # $/million output tokens
    cache_read: float = 0.0
    cache_write: float = 0.0

class ModelRegistry:
    def register(self, model: ModelDefinition) -> None
    def get(self, model_id: str) -> ModelDefinition | None
    def find(self, query: str) -> list[ModelDefinition]  # Fuzzy match
    def list_by_provider(self, provider: str) -> list[ModelDefinition]
    def calculate_cost(self, model_id: str, usage: TokenUsage) -> float
```

**Includes**:
- Built-in catalog for major providers (generated or maintained as YAML/JSON)
- `TokenUsage` tracking integrated into adapter responses
- `ModelRegistry` injectable via config, extensible at runtime

**Files to create/modify**:
- `src/skillengine/model_registry.py` — `ModelDefinition`, `ModelRegistry`, `ModelCost`
- `src/skillengine/models_catalog.py` — Built-in model definitions (or `models_catalog.yaml`)
- `src/skillengine/adapters/base.py` — Add `TokenUsage` to `AgentResponse`
- `src/skillengine/config.py` — Accept `ModelDefinition` in `AgentConfig`

---

## P1 — Context Management Pipeline

**Problem**: No context window management. Long conversations will exceed model limits and fail. No mechanism to compress, prune, or transform messages before sending to LLM.

**Current state**: Messages accumulate unbounded. No awareness of token count or context window.

**Target design**:

```python
class AgentRunner:
    async def chat(self, message: str) -> AgentResponse:
        # 1. Append user message
        self.messages.append(user_msg)

        # 2. Transform context (hook for pruning/injection)
        context = await self.transform_context(self.messages)

        # 3. Convert to LLM format (filter agent-only messages)
        llm_messages = self.convert_to_llm(context)

        # 4. Check token budget, compact if needed
        if self.estimate_tokens(llm_messages) > model.context_window * 0.9:
            llm_messages = await self.compact(llm_messages)

        # 5. Send to LLM
        ...
```

**Key components**:

```python
# Message type separation
class AgentMessage:
    """Superset — includes UI-only messages, metadata, etc."""

class LLMMessage:
    """Strict LLM format — user/assistant/tool_result only."""

# Context transformation hook
ContextTransformer = Callable[[list[AgentMessage]], Awaitable[list[AgentMessage]]]

# Built-in compaction strategies
class SlidingWindowCompactor:
    """Keep last N turns, summarize older ones."""

class TokenBudgetCompactor:
    """Prune oldest messages to fit within token budget."""
```

**Files to create/modify**:
- `src/skillengine/context.py` — `ContextTransformer`, compaction strategies, token estimation
- `src/skillengine/agent.py` — Integrate context pipeline into agent loop
- `src/skillengine/models.py` — Distinguish `AgentMessage` vs `LLMMessage`

---

## P2 — Tool Execution Streaming

**Problem**: Tool execution is fire-and-forget. Long-running commands (bash scripts, API calls) give no feedback until completion.

**Current state**:

```python
class SkillRuntime(ABC):
    async def execute(self, command, cwd, env, timeout) -> ExecutionResult
    # Returns only after completion — no intermediate output
```

**Target design**:

```python
class SkillRuntime(ABC):
    async def execute(
        self,
        command: str,
        cwd: Path | None = None,
        env: dict | None = None,
        timeout: float | None = None,
        on_output: Callable[[str], None] | None = None,  # NEW: streaming callback
    ) -> ExecutionResult:
        ...

# BashRuntime streams stdout/stderr line by line
class BashRuntime(SkillRuntime):
    async def execute(self, command, cwd, env, timeout, on_output=None):
        proc = await asyncio.create_subprocess_shell(...)
        async for line in proc.stdout:
            if on_output:
                on_output(line.decode())
        ...
```

**Integration with event system**:

```python
# Agent emits tool_execution_update events
@agent.events.on("tool_execution_update")
def on_tool_output(event):
    print(f"[{event.tool_name}] {event.output}", end="")
```

**Files to modify**:
- `src/skillengine/runtime/base.py` — Add `on_output` parameter
- `src/skillengine/runtime/bash.py` — Stream subprocess output
- `src/skillengine/agent.py` — Emit `tool_execution_update` events

---

## P2 — Steering & Abort

**Problem**: Once the agent loop starts, the caller has no control. Cannot cancel a runaway tool, interrupt a multi-turn loop, or inject a correction mid-execution.

**Current state**: `chat()` blocks until the full agent loop completes.

**Target design**:

```python
class AgentRunner:
    async def chat(self, message: str) -> AgentResponse:
        ...

    def abort(self) -> None:
        """Cancel current operation immediately."""
        self._abort_controller.abort()

    async def steer(self, message: str) -> None:
        """Interrupt current tool chain, inject new instruction.
        The agent will stop executing remaining tools and process
        this message in the next turn."""
        self._steering_queue.put(message)

    async def follow_up(self, message: str) -> None:
        """Queue a message to send after the current agent loop ends.
        Triggers a new agent loop with this message."""
        self._followup_queue.put(message)
```

**Implementation notes**:
- Pass `AbortSignal`-equivalent (e.g., `asyncio.Event`) to tool execution and LLM streaming
- Check steering queue between tool executions in the agent loop
- Follow-up queue checked after agent loop completes, triggering re-entry

**Files to modify**:
- `src/skillengine/agent.py` — Add `abort()`, `steer()`, `follow_up()`, internal queues
- `src/skillengine/runtime/bash.py` — Respect abort signal in subprocess execution
- `src/skillengine/adapters/base.py` — Respect abort signal in LLM streaming

---

## P3 — Dynamic Provider Registry

**Problem**: Adapters are bound at initialization. Cannot add a new LLM provider at runtime (e.g., from an extension or plugin).

**Current state**: Single adapter instance passed to `AgentRunner`.

**Target design**:

```python
class AdapterRegistry:
    def register(self, name: str, adapter: LLMAdapter, source: str | None = None) -> None
    def unregister(self, name: str) -> None
    def unregister_by_source(self, source: str) -> None
    def get(self, name: str) -> LLMAdapter
    def list(self) -> list[str]

# Extension registers a custom provider at runtime
class MyExtension:
    def activate(self, engine):
        engine.adapters.register(
            "my-local-llm",
            OpenAIAdapter(engine, base_url="http://localhost:8080/v1"),
            source="my-extension",
        )

# Switch provider mid-session
agent.set_adapter("my-local-llm")
```

**Files to create/modify**:
- `src/skillengine/adapters/registry.py` — `AdapterRegistry`
- `src/skillengine/agent.py` — Use registry instead of single adapter
- `src/skillengine/extensions/api.py` — Expose `register_adapter()` to extensions

---

## Implementation Order

```
Phase 1 (Foundation)
  ├── P0: Event System
  └── P0: Structured Stream Events

Phase 2 (Intelligence)
  ├── P1: Model Metadata & Registry
  └── P1: Context Management Pipeline

Phase 3 (Control)
  ├── P2: Tool Execution Streaming
  └── P2: Steering & Abort

Phase 4 (Extensibility)
  └── P3: Dynamic Provider Registry
```

Each phase builds on the previous:
- Phase 1 enables Phase 2 (context management uses events for `context_transform`)
- Phase 1 enables Phase 3 (tool streaming emits events; steering checks events between tools)
- Phase 2 enables Phase 4 (registry needs model metadata to resolve adapters)
