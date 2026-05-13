# Reference

Public API, grouped by area. Every name listed here is importable directly from
the top-level `skillengine` package unless noted.

## Core

| Symbol | Purpose |
|---|---|
| `SkillsEngine` | Loads, filters, snapshots, and executes skills |
| `SkillsConfig` | Engine configuration (skill dirs, per-skill overrides) |
| `SkillEntryConfig` | Per-skill enable/disable/override knobs |
| `CacheRetention` | Snapshot cache policy |
| `Skill`, `SkillMetadata`, `SkillRequirements` | Core dataclasses |
| `SkillSnapshot` | Immutable view returned by `engine.get_snapshot()` |

## Agent

| Symbol | Purpose |
|---|---|
| `AgentRunner` | Conversation orchestrator |
| `AgentConfig` | Model, base URL, API key, max turns, tool flags |
| `AgentMessage` | A single message in the conversation log |
| `TextContent`, `ImageContent` | Multi-modal message parts |

Key methods: `chat`, `chat_structured`, `chat_stream_events`,
`connect_mcp_servers`, `disconnect_mcp_servers`, `add_handoffs`,
`remove_handoffs`, `switch_model`.

## Events

`EventBus` with handler registration via `on(event_type, handler, priority=0)`.
Event types live in `skillengine.events`.

`BEFORE_TOOL_CALL` handlers can return `ToolCallEventResult(block=True)` to
cancel a tool call.

## Tools

`ToolDispatcher` registers `ToolDefinition` objects via
`dispatcher.register(name, handler, definition)` where `handler` has the shape
`async (name, args, ctx, on_output) -> str` and `definition` is an OpenAI
function-calling object.

Built-in tools: `BashTool`, `ReadTool`, `WriteTool`, `EditTool`, `GrepTool`,
`FindTool`, `LsTool`.

## Adapters

| Symbol | Purpose |
|---|---|
| `LLMAdapter` | Abstract base |
| `OpenAIAdapter` | OpenAI + OpenAI-compatible |
| `AnthropicAdapter` | Claude with adaptive thinking |
| `AdapterRegistry`, `AdapterFactory` | Lazy instantiation |

## MCP

| Symbol | Purpose |
|---|---|
| `MCPClient` | JSON-RPC stdio client (`tools/list`, `tools/call`, …) |
| `MCPServer` | Server side, exposes skills as `skill__<name>` |
| `MCPServerSpec` | Transport, command, args, env, cwd |
| `MCPConnectionPool` | Manages multiple `MCPClient`s + dispatcher integration |
| `parse_mcp_uri(uri)` | Parse `mcp+stdio:…`, `mcp+stdio://…`, `mcp+command:{…}` |

## A2A / Handoffs

| Symbol | Purpose |
|---|---|
| `AgentCard`, `AgentCardSkill`, `AgentCapabilities` | A2A protocol cards |
| `AgentRegistry`, `RegisteredAgent`, `AgentSource`, `AgentStats` | Registry |
| `A2AClient` | Send tasks to remote A2A agents |
| `AgentDiscovery`, `DiscoveryConfig`, `AgentHealthStatus` | Auto-discovery |
| `PerformanceRouter`, `RoutingConfig`, `RouteResult` | Latency-aware routing |
| `CoordinatorAgent`, `CoordinatorConfig` | Multi-agent orchestrator |
| `Handoff`, `handoff`, `callable_handoff`, `agent_handoff`, `a2a_handoff` | OpenAI Agents / Anthropic A2A compat shim |

## Structured output

| Symbol | Purpose |
|---|---|
| `extract_json_schema(T)` | JSON schema for `T` (Pydantic / dataclass / TypedDict / dict) |
| `extract_json_payload(text)` | Pull a JSON value out of arbitrary text |
| `parse_structured(T, text)` | Combine extract + validate |
| `build_directive(T)` | Prompt suffix appended by `chat_structured` |
| `StructuredOutputError` | Raised on unrecoverable parse / validation failures |

## CLI

```
uv run skills list -d ./skills        # list eligible skills
uv run skills show <name>             # print a skill's body
uv run skills validate -d ./skills    # validate names/descriptions
uv run skills prompt -d ./skills      # print the system prompt
uv run skills exec <name> ...         # execute a skill's commands
uv run skills watch -d ./skills       # hot-reload
```

```
python -m skillengine.mcp --skill-dir ./skills [--name N] [--version V]
                                              [--instructions S]
```

## Runtimes

| Symbol | Purpose |
|---|---|
| `BashRuntime` | Shell execution with timeout + streaming |
| `CodeModeRuntime` | `search` + `execute` pattern over an arbitrary spec |
| `BoxLiteRuntime` (optional) | VM-level isolation via micro-VMs |
