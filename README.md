<p align="center">
  <h1 align="center">SkillEngine</h1>
  <p align="center">A framework-agnostic skills execution engine for LLM agents</p>
</p>

<p align="center">
  <a href="https://github.com/sawzhang/skillengine/actions/workflows/ci.yml"><img src="https://github.com/sawzhang/skillengine/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/skillengine/"><img src="https://img.shields.io/pypi/v/skillengine" alt="PyPI"></a>
  <a href="https://pypi.org/project/skillengine/"><img src="https://img.shields.io/pypi/pyversions/skillengine" alt="Python"></a>
  <a href="https://github.com/sawzhang/skillengine/blob/master/LICENSE"><img src="https://img.shields.io/github/license/sawzhang/skillengine" alt="License"></a>
</p>

---

Build Claude Code-like experiences with automatic skill discovery, on-demand loading, tool execution, and multi-provider support.

```bash
pip install skillengine
```

## Highlights

- **Progressive Disclosure** — Only skill names + descriptions in the system prompt; full content loaded on-demand via the `skill` tool
- **Multi-Provider** — OpenAI, Anthropic, Google, DeepSeek, MiniMax + any OpenAI-compatible endpoint
- **Plugin Everything** — Loaders, filters, runtimes, adapters, extensions — all swappable
- **Production Ready** — Sessions, context compaction, cost tracking, prompt caching, sandbox isolation, event hooks, hot-reload

## Quick Start

```python
import asyncio
from pathlib import Path
from skillengine import create_agent

async def main():
    agent = await create_agent(
        skill_dirs=[Path("./skills")],
        system_prompt="You are a helpful assistant.",
    )

    response = await agent.chat("Help me create a PDF report")
    print(response.content)

    # Slash commands invoke skills directly
    response = await agent.chat("/pdf extract text from invoice.pdf")
    print(response.content)

asyncio.run(main())
```

## Installation

```bash
pip install skillengine                # Core
pip install skillengine[openai]        # + OpenAI adapter
pip install skillengine[anthropic]     # + Anthropic adapter
pip install skillengine[openai,anthropic,websockets,memory,web,sandbox]  # Everything
```

## Architecture

```
                        ┌─────────────────────────────┐
                        │         AgentRunner          │
                        │  on-demand skills · slash    │
                        │  commands · fork isolation   │
                        │  per-skill model & tools     │
                        └──────────┬──────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │        SkillsEngine          │
                        │                              │
                        │  Loader → Filter → Runtime   │
                        │         ↓                    │
                        │    SkillSnapshot             │
                        └──────────┬──────────────────┘
                                   │
          ┌────────┬───────┬───────┼───────┬──────────┐
          │        │       │       │       │          │
       OpenAI  Anthropic Google DeepSeek MiniMax   Custom
```

| Layer | Base Class | Built-in | Purpose |
|-------|-----------|----------|---------|
| Loader | `SkillLoader` | `MarkdownSkillLoader` | Parse SKILL.md (Markdown + YAML frontmatter) |
| Filter | `SkillFilter` | `DefaultSkillFilter` | Eligibility: bins, env vars, OS, config |
| Runtime | `SkillRuntime` | `BashRuntime` `CodeModeRuntime` `BoxLiteRuntime` | Execute with timeout, streaming, sandboxing |
| Adapter | `LLMAdapter` | `OpenAIAdapter` `AnthropicAdapter` | Provider integration with tool calling |

## Defining Skills

Create `skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: "Does useful things"
model: claude-sonnet-4-20250514   # Per-skill model override
context: fork                      # Isolated subagent
allowed-tools: [Read, Grep, Glob]  # Tool restrictions
user-invocable: true               # Enable /my-skill slash command
argument-hint: "<query>"

metadata:
  emoji: "🔧"
  primary_env: "API_KEY"
  requires:
    bins: ["git"]              # ALL must exist
    any_bins: ["npm", "pnpm"]  # At least ONE
    env: ["GITHUB_TOKEN"]
    os: ["darwin", "linux"]
---

# My Skill

Instructions for the LLM when this skill is loaded.

Process: $ARGUMENTS
Current branch: !`git branch --show-current`
```

| Placeholder | Replaced with |
|---|---|
| `$ARGUMENTS` | Full argument string |
| `$1` `$2` ... `$N` | Positional arguments |
| `${CLAUDE_SESSION_ID}` | Session ID |
| `` !`cmd` `` | Command stdout (before sending to LLM) |

## Core API

### AgentRunner (high-level)

```python
from skillengine import AgentRunner, AgentConfig, create_agent

config = AgentConfig(
    model="gpt-4o",
    api_key="...",
    max_turns=20,
    thinking_level="medium",        # off / minimal / low / medium / high / xhigh
    skill_description_budget=16000, # Max chars for skill metadata in system prompt
)
agent = AgentRunner(engine, config)

response = await agent.chat("Hello")                        # Single turn
async for chunk in agent.chat_stream("Explain this"):       # Streaming
    print(chunk, end="")
await agent.run_interactive()                                # REPL
```

### SkillsEngine (low-level)

```python
from skillengine import SkillsEngine, SkillsConfig

engine = SkillsEngine(config=SkillsConfig(
    skill_dirs=[Path("./skills")],
    watch=True,
    prompt_format="xml",  # xml / markdown / json
))

snapshot = engine.get_snapshot()
print(snapshot.prompt)                           # Inject into your system prompt
result = await engine.execute("echo hello")      # Execute a command
```

## Key Features

### Model Registry & Cost Tracking

Built-in catalog for 12+ models across 5 providers with pricing, context windows, and capabilities.

```python
from skillengine import ModelRegistry, TokenUsage

registry = ModelRegistry()
model = registry.get("gpt-4o")
# model.context_window → 128000
# model.cost.input → 2.5 ($/M tokens)

cost = registry.calculate_cost("gpt-4o", TokenUsage(input_tokens=1000, output_tokens=500))
```

### Extended Thinking

Maps thinking budgets across providers transparently.

```python
AgentConfig(model="claude-opus-4-20250514", thinking_level="high")
```

| Level | Anthropic budget | OpenAI effort |
|-------|:---:|:---:|
| minimal | 1,024 | low |
| low | 2,048 | low |
| medium | 4,096 | medium |
| high | 8,192 | high |
| xhigh | 16,384 | high |

### Events

13 lifecycle events for observability and control flow.

```python
from skillengine import EventBus, BEFORE_TOOL_CALL

bus = EventBus()

@bus.on(BEFORE_TOOL_CALL)
async def guard(event):
    if "rm -rf" in event.arguments.get("command", ""):
        return ToolCallEventResult(block=True, message="Blocked")
```

Events: `AGENT_START` `AGENT_END` `TURN_START` `TURN_END` `BEFORE_TOOL_CALL` `AFTER_TOOL_RESULT` `INPUT` `CONTEXT_TRANSFORM` `TOOL_EXECUTION_UPDATE` `SESSION_START` `SESSION_END` `MODEL_CHANGE` `COMPACTION`

### Context Compaction

```python
from skillengine import TokenBudgetCompactor, SlidingWindowCompactor

compactor = TokenBudgetCompactor(max_tokens=100_000)  # Token budget
compactor = SlidingWindowCompactor(window_size=50)     # Keep N recent messages
```

### Sessions

JSONL append-only tree with branching.

```python
from skillengine.session import SessionManager

mgr = SessionManager(store_dir=Path("./sessions"))
session = mgr.create_session()
branch = mgr.branch(session.id, entry_index=5)  # Fork from earlier point
```

### CodeModeRuntime

Cloudflare code-mode-mcp pattern: 2 tools (`search` + `execute`) instead of N. Token cost O(1).

```python
from skillengine import CodeModeRuntime

runtime = CodeModeRuntime(spec=openapi_spec, ctx={"client": httpx.Client()})
await runtime.search("[p for p in spec['paths'] if '/users' in p]")
await runtime.run("result = ctx['client'].get('/users')")
```

### Sandbox Execution

VM-level isolation via BoxLite micro-VMs.

```python
from skillengine import BoxLiteRuntime, SecurityLevel, SandboxedAgentRunner

runtime = BoxLiteRuntime(security_level=SecurityLevel.STANDARD)
agent = SandboxedAgentRunner(engine, config)  # Full agent in sandbox
```

### Built-in Tools

| Tool | Purpose | Tool | Purpose |
|------|---------|------|---------|
| `bash` | Shell commands | `read` | Read files |
| `write` | Write files | `edit` | Text replacement |
| `grep` | Pattern search | `find` | File discovery |
| `ls` | Directory listing | `skill` | On-demand skill loading |

### Extensions

```python
# ~/.skillengine/extensions/my_ext.py
def extension(api):
    api.on("turn_end", my_hook)
    api.register_command("/my-cmd", handler, description="...")
    api.register_tool("my_tool", schema, handler)
    api.register_adapter("my-provider", MyAdapter)
```

### Execution Modes

| Mode | Protocol | Use case |
|------|----------|----------|
| Interactive | TUI (stdin/stdout) | Terminal REPL |
| JSON | JSONL to stdout | Pipelines |
| RPC | JSON-line stdin/stdout | Programmatic control |

### CLI

```bash
skills list -d ./skills           # List skills
skills show pdf -d ./skills       # Show detail
skills prompt -d ./skills -f xml  # Generate prompt
skills exec "echo hello"          # Run command
skills validate -d ./skills       # Validate
skills watch -d ./skills          # Watch + hot-reload
```

## Extending

Every layer is pluggable. Implement the base class and pass to the engine:

```python
from skillengine.loaders import SkillLoader
from skillengine.filters import SkillFilter
from skillengine.runtime import SkillRuntime

class MyLoader(SkillLoader):       # Custom file format
    def can_load(self, path): ...
    def load_skill(self, path, source): ...

class MyFilter(SkillFilter):       # Custom eligibility
    def filter(self, skill, config, context): ...

class MyRuntime(SkillRuntime):     # Custom execution
    async def execute(self, command, cwd, env, timeout): ...
```

## Development

```bash
git clone https://github.com/sawzhang/skillengine.git
cd skillengine
uv sync

uv run pytest                # 1083 tests
uv run ruff check src/       # Lint
uv run ruff format --check src/  # Format check
uv run mypy src/             # Type check
```

CI runs on every push and PR: lint + format + mypy on Python 3.12, tests on 3.10/3.11/3.12.

## License

MIT
