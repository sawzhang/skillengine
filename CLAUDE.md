# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkillKit is a framework-agnostic skills execution engine for LLM agents. It provides a plugin-based system for defining, loading, filtering, and executing skills (capabilities) that can be made available to large language models. Aligned with the Claude Agent Skills architecture (progressive disclosure, on-demand loading, per-skill model/tools).

## Commands

```bash
# Install dependencies
uv sync

# Install with optional adapters
uv add -e ".[openai]"
uv add -e ".[anthropic]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_engine.py

# Run a specific test
pytest tests/test_engine.py::test_function_name -v

# Linting and formatting
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## Architecture

The engine uses a pipeline architecture with four extensible subsystems:

```
Skill Files (Markdown+YAML) → [Loader] → [Filter] → [Runtime] → [Adapter]
```

### Core Components

- **Engine** (`engine.py`): Orchestrates the entire pipeline - loads skills from directories, applies filters, generates prompts, and executes commands
- **Agent** (`agent.py`): `AgentRunner` with on-demand skill loading via `skill` tool, `$ARGUMENTS` substitution, `context: fork`, `!`cmd`` dynamic injection, per-skill model switching
- **Models** (`models.py`): Data structures including `Skill`, `SkillMetadata`, `SkillRequirements`, `SkillSnapshot`, `AgentPersona`
- **Config** (`config.py`): `SkillsConfig` for directory management, filtering options, per-skill overrides

### Plugin Subsystems

Each subsystem has an abstract base class and reference implementation:

| Subsystem | Base Class | Implementation | Purpose |
|-----------|------------|----------------|---------|
| Loaders | `SkillLoader` | `MarkdownSkillLoader` | Parse skill files (Markdown + YAML frontmatter) |
| Filters | `SkillFilter` | `DefaultSkillFilter` | Check eligibility (bins, env vars, OS, config) |
| Runtimes | `SkillRuntime` | `BashRuntime`, `CodeModeRuntime` | Execute commands with timeout and env injection |
| Adapters | `LLMAdapter` | `OpenAIAdapter`, `AnthropicAdapter` | Integrate with LLM providers |

### Skill Definition Format

Skills are defined as Markdown files with YAML frontmatter, located at `skills/<name>/SKILL.md`:

```yaml
---
name: skill-name
description: What the skill does
model: claude-sonnet-4-5-20250514    # Per-skill model override
context: fork                  # Isolated subagent execution
allowed-tools: [Read, Grep]   # Tool restrictions
argument-hint: "<query>"       # Autocomplete hint
hooks:
  PreToolExecution: "echo pre"
metadata:
  emoji: "🔧"
  primary_env: "API_KEY"
  memory_scope: "code-review" # Isolated memory namespace
  requires:
    bins: ["git"]           # ALL must exist
    any_bins: ["npm", "pnpm"] # At least ONE must exist
    env: ["GITHUB_TOKEN"]
    os: ["darwin", "linux"]
---
# Skill content (prompt text)
Use $ARGUMENTS for dynamic input.
Current date: !`date +%Y-%m-%d`
```

### On-Demand Skill Loading

The system prompt only contains skill **names and descriptions** (metadata). Full skill content is loaded on-demand when the LLM calls the `skill` tool. This follows the progressive disclosure pattern from Claude Agent Skills.

- `AgentConfig.skill_description_budget` (default 16000) limits total chars for skill metadata in the system prompt
- The `skill` tool accepts `name` and optional `arguments` parameters
- `$ARGUMENTS`, `$1`..`$N`, `${CLAUDE_SESSION_ID}` are substituted in skill content
- `!`command`` placeholders are replaced with command stdout before sending to LLM
- Skills with `context: fork` run in an isolated child `AgentRunner`
- Skills with `model:` trigger per-skill model switching (restored after)
- Skills with `allowed-tools:` restrict which tools the LLM can use

### Skill Validation

`AgentRunner.validate_skill(skill)` enforces:
- Name: ≤64 chars, lowercase alphanumeric + hyphens, no leading hyphen
- Description: non-empty, ≤1024 chars

### AgentPersona (Soul Layer)

`AgentPersona` provides structured agent identity instead of embedding persona as free-form text in `system_prompt`. Rendered as the first section in the system prompt.

```python
persona = AgentPersona(
    name="CodeReviewer",
    role="You are a senior code reviewer.",
    style="Be concise and direct. Use bullet points.",
    constraints=["Never approve code with security vulnerabilities"],
)
config = AgentConfig(persona=persona, system_prompt="Base instructions.")
```

**SOUL.md auto-discovery**: If no programmatic `persona` is set on `AgentConfig`, the agent auto-discovers `SOUL.md` files (cwd → parents → `~/.skillkit/`). SOUL.md supports optional YAML frontmatter:

```yaml
---
name: CodeBot
style: friendly and encouraging
constraints:
  - always explain your reasoning
---
You are a coding tutor for beginners.
```

### Memory Scope

Skills can declare an isolated memory namespace via `metadata.memory_scope` in the SKILL.md frontmatter. During skill execution, the memory tools (`recall_memory`, `save_memory`) automatically use this scope instead of the default `"user"` scope. The scope is pushed on skill entry and popped on exit (including on error).

- `MemoryConfig.default_memory_scope` sets the global default (default: `"user"`)
- `SkillMetadata.memory_scope` overrides per-skill
- `MemoryState.push_scope()` / `pop_scope()` manage the active scope stack
- `save_memory` annotates with `[memory:category:scope]` format

### Task Scheduler

`TaskScheduler` enables agent-initiated behavior — periodic tasks, one-shot delays, and event-triggered actions. Integrates with `EventBus` and can execute skills by name.

```python
from skillkit.scheduler import TaskScheduler, ScheduledTask, TaskTrigger

# Via AgentRunner (recommended)
scheduler = runner.setup_scheduler()

# Or standalone
scheduler = TaskScheduler(event_bus=runner.events, skill_executor=my_executor)

# Interval task — runs every 60s
scheduler.add(ScheduledTask(
    name="health-check",
    trigger=TaskTrigger.INTERVAL,
    action=check_health,
    interval_seconds=60,
))

# One-shot — runs once after 5s delay
scheduler.add(ScheduledTask(
    name="init",
    trigger=TaskTrigger.ONCE,
    action=initialize,
    delay_seconds=5,
))

# Event-triggered — fires when EventBus emits "turn_end"
scheduler.add(ScheduledTask(
    name="on-turn-end",
    trigger=TaskTrigger.EVENT,
    action=summarize,
    event_name="turn_end",
))

# Skill execution — invoke a skill by name
scheduler.add(ScheduledTask(
    name="periodic-review",
    trigger=TaskTrigger.INTERVAL,
    skill_name="code-review",
    skill_args="check for issues",
    interval_seconds=300,
))

await scheduler.start()
# ... agent runs ...
await scheduler.stop()
```

- **Trigger types**: `INTERVAL`, `ONCE`, `EVENT`, `CRON` (simple `*/N` minute expressions)
- **Concurrency control**: `SchedulerConfig.max_concurrent` limits parallel task executions
- **Error resilience**: consecutive errors → cooldown → auto-disable after `max_errors`
- **Task lifecycle**: `add()`, `remove()`, `pause()`, `resume()`, `get()`, `tasks`, `active_tasks`
- **AgentRunner integration**: `runner.setup_scheduler()` creates a scheduler with skill execution wired in; auto-starts on first `chat()` call

### CodeModeRuntime (search + execute pattern)

Inspired by Cloudflare's code-mode-mcp. Instead of exposing N tools (one per API endpoint), `CodeModeRuntime` exposes just 2 tools — `search` and `execute` — and lets the LLM write Python code against injected data and clients. Token cost is O(1) regardless of API surface area.

```python
runtime = CodeModeRuntime(
    spec=openapi_spec,              # Any data for discovery (dict, list, etc.)
    ctx={"client": httpx.Client()}, # Objects injected into execute mode
)

# search: spec only, no ctx (read-only discovery)
await runtime.search("[p for p in spec['paths'] if '/users' in p]")

# run/execute: spec + ctx (call APIs, mutate state)
await runtime.run("result = ctx['client'].get('/users')")

# Generate 2-tool definitions for LLM adapters (OpenAI format)
tools = runtime.get_tool_definitions()  # → [search, execute]
```

- **Two execution modes**: `search(code)` injects `spec` only; `run(code)` / `execute(code)` injects both `spec` and `ctx`
- **Two sandbox modes**: `"inprocess"` (exec with restricted builtins, works with any Python objects) and `"subprocess"` (child process isolation, JSON-serializable data only)
- **Safe builtins**: restricted `__import__` only allows configured modules (default: json, re, math, datetime, collections, itertools, functools, urllib.parse)
- **`get_tool_definitions()`**: generates OpenAI function-calling format tool definitions with spec hints and ctx key names
- **Drop-in replacement**: implements `SkillRuntime`, can be passed to `SkillsEngine(runtime=CodeModeRuntime(...))`
- **Result convention**: user code assigns to `result` for structured output, or uses `print()` for text output

### A2A (Agent-to-Agent) Integration

The `skillkit.a2a` module provides agent discovery, registration, and routing:

- **AgentCard** (`a2a/agent_card.py`): Auto-generated from Skill via `AgentCard.from_skill()`. Includes `to_dict()` for A2A JSON, `to_embedding_text()` for semantic indexing, `to_summary_line()` for system prompt injection.
- **AgentRegistry** (`a2a/registry.py`): Unified registry for local (Skill-based) and remote (A2A) agents. `cards_summary(budget)` generates token-budgeted summaries. `awareness_prompt_block()` returns a complete system prompt section. `match(query, top_k)` does keyword routing.
- **A2A Models** (`a2a/models.py`): `A2ATaskRequest`, `A2ATaskResponse`, `TaskStatus` for protocol interop.
- **A2A Server** (`a2a/server.py`): FastAPI app exposing `/.well-known/agent.json`, `/tasks`, `/tasks/{id}`, `/tasks/{id}/cancel`, `/health`. Optional dep: `pip install fastapi uvicorn`.
- **A2A Client** (`a2a/client.py`): `discover()` fetches remote Agent Cards, `send_task()` invokes remote agents, `discover_and_register()` combines both. `create_remote_agent_tool()` generates LLM tool definition for Orchestrator routing.
- **Claude SDK Bridge** (`a2a/claude_sdk_bridge.py`): `run_skill()` executes Skills via `claude_agent_sdk.query()`, `to_sdk_agent_definition()` exports Skills as SDK sub-agents, `to_mcp_tool()` exports as MCP tool definitions. Optional dep: `pip install claude-agent-sdk`.
- **Discovery** (`a2a/discovery.py`): `AgentDiscovery` manages automatic discovery and health monitoring. Periodic endpoint scanning, health checking with configurable failure thresholds, auto-removal of unhealthy agents, event emission (`AGENT_DISCOVERED`, `AGENT_REMOVED`, `AGENT_HEALTH_CHANGED`). Config via `DiscoveryConfig`.
- **Performance Router** (`a2a/router.py`): `PerformanceRouter` scores agents on keyword relevance (40%), performance history (30%), cost (15%), and recency (15%). Supports fallback chains via `route_with_fallback()`, outcome recording for learning, and `routing_report()` for debugging. Config via `RoutingConfig`.
- **Loader Integration**: `MarkdownSkillLoader` parses optional `a2a:` frontmatter block and attaches as `skill._a2a_config`.

SKILL.md `a2a:` frontmatter extension (optional):
```yaml
a2a:
  expose: true
  input_schema: { type: object, properties: { ... } }
  output_schema: { type: object, properties: { ... } }
  stateful: false
  cost_hint: low  # low / medium / high
```

### Key Patterns

- **Progressive Disclosure**: Only skill metadata in system prompt; full content loaded on-demand via `skill` tool
- **Environment Management**: The engine uses a context manager pattern (`env_context`) to safely backup/restore environment variables when injecting per-skill overrides
- **Snapshot Caching**: `SkillSnapshot` provides immutable point-in-time views with version tracking and content hashing for cache invalidation
- **Filter Short-circuiting**: Eligibility checks run in sequence and short-circuit on first failure, returning the reason for ineligibility
- **Per-Skill Model Switching**: `switch_model()` before skill content, restore in `finally` block
- **Fork Isolation**: `_execute_skill_forked()` creates a child `AgentRunner` with skill content as system prompt, inherits parent config
