# Agents

`AgentRunner` is the orchestrator. It owns a `SkillsEngine`, a tool dispatcher,
an event bus, and the conversation state.

## Lifecycle

```python
from skillengine import AgentConfig, AgentRunner, SkillsConfig, SkillsEngine

engine = SkillsEngine(SkillsConfig(skill_dirs=["./skills"]))
runner = AgentRunner(
    engine=engine,
    config=AgentConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
    ),
)
message = await runner.chat("Help me with X")
```

## What happens on each turn

1. **System prompt build** — names + descriptions of eligible skills (budgeted by
   `skill_description_budget`, default 16000 chars).
2. **Adapter call** — message history sent to the LLM with the tool catalogue.
3. **Tool execution** — `ToolDispatcher` runs each call returned by the LLM and
   appends results to the history.
4. **Skill loading** — when the LLM calls the `skill` tool, the full body of the
   requested skill is injected into the next turn.
5. **Repeat** until the LLM produces a final assistant message or `max_turns` is
   reached.

## Events

`EventBus` exposes 13 event types:

| Category | Events |
|---|---|
| Lifecycle | `AGENT_START`, `AGENT_END`, `SESSION_START`, `SESSION_END` |
| Turn | `TURN_START`, `TURN_END` |
| Tool | `BEFORE_TOOL_CALL` *(blockable)*, `AFTER_TOOL_RESULT`, `TOOL_EXECUTION_UPDATE` |
| Context | `CONTEXT_TRANSFORM`, `INPUT`, `COMPACTION` |
| Model | `MODEL_CHANGE` |

A `BEFORE_TOOL_CALL` handler can return `ToolCallEventResult(block=True)` to veto
a call.

## Multi-agent

- **`connect_mcp_servers([spec, ...])`** — pull tools from remote MCP servers.
- **`add_handoffs([handoff, ...])`** — register `transfer_to_<agent>` tools that
  delegate to other agents (local, remote A2A, or any async callable).

See [`interop.md`](./interop.md) for the protocol details.
