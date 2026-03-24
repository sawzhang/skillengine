# SkillEngine

**The open-source skills engine that gives any LLM agent a Claude Code-like experience.**

Define skills as Markdown. Load them into any model. Ship agent capabilities without vendor lock-in.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Why SkillEngine?

Every AI agent platform is building its own skills/plugins system — Claude Code has skills, Cursor has rules, ChatGPT has GPTs, Codex has tasks. They're all incompatible, proprietary, and locked to one provider.

SkillEngine extracts the **best patterns from Claude Code's skill system** into a standalone, framework-agnostic engine that works with any LLM. Write once, run on OpenAI, Anthropic, MiniMax, or your local model.

```
Your Skills (Markdown + YAML)
        │
        ▼
   ┌──────────┐
   │ SkillEngine │ ← Framework-agnostic engine
   └────┬─────┘
        │
   ┌────┴────────────────────────┐
   │         │         │         │
 OpenAI  Anthropic  MiniMax   Local
```

## What You Get

- **Markdown-based skills** — Define agent capabilities as simple `SKILL.md` files with YAML frontmatter
- **On-demand loading** — Only skill names/descriptions in system prompt; full content loaded when the LLM needs it
- **Slash commands** — `/pdf`, `/pptx`, etc. — user-invocable skills, just like Claude Code
- **Per-skill model & tools** — Each skill can specify its own model and allowed tools
- **Context fork** — Run skills in isolated subagent contexts
- **Dynamic injection** — `$ARGUMENTS`, `$1`..`$N` substitution + `` !`command` `` shell expansion
- **Eligibility filtering** — Auto-filter skills by OS, binaries, env vars, and config
- **Hot-reload** — File watcher reloads skills on save, no restart needed
- **Multi-source** — Load from bundled, managed, workspace, and plugin directories
- **Validation** — Enforces naming rules, description limits, and metadata schemas

## Quick Start

### Install

```bash
# With uv (recommended)
uv add skillengine

# Or pip
pip install skillengine[openai]
```

### Create a Skill

Create `skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: "Summarize any document into bullet points"
user-invocable: true
allowed-tools:
  - Read
  - Write
---

# Document Summarizer

Read the file at $ARGUMENTS and produce a concise bullet-point summary.
Focus on key decisions, action items, and open questions.
```

### Run the Agent

```python
import asyncio
from pathlib import Path
from skillengine import create_agent

async def main():
    agent = await create_agent(
        skill_dirs=[Path("./skills")],
        system_prompt="You are a helpful assistant.",
        watch_skills=True,
    )

    # Natural language — LLM loads skills on demand
    response = await agent.chat("Help me create a PDF report")
    print(response.content)

    # Slash command — direct skill invocation
    response = await agent.chat("/pdf extract text from invoice.pdf")
    print(response.content)

asyncio.run(main())
```

### Interactive Mode

```bash
uv run python examples/agent_demo.py --interactive
```

Commands: `/skills` (list all), `/pdf`, `/pptx` (invoke skill), `/clear`, `/quit`

## How Skills Work

```
System Prompt (lightweight)              On-Demand Loading (full content)
┌─────────────────────────┐             ┌──────────────────────────────┐
│ Available skills:       │   LLM calls │ SKILL.md full content        │
│ - pdf: PDF extraction   │ ──────────► │ + $ARGUMENTS substituted     │
│ - pptx: Slide creation  │  skill()    │ + !`commands` expanded       │
│ - my-skill: Summarizer  │   tool      │ + env vars injected          │
└─────────────────────────┘             └──────────────────────────────┘
```

Only names and descriptions live in the system prompt (configurable budget: default 16K chars). The LLM calls `skill(name="pdf", arguments="report.pdf")` to load full content when needed — **progressive disclosure** that keeps context lean.

## Skill Metadata Reference

```yaml
---
name: skill-name                   # ≤64 chars, lowercase + digits + hyphens
description: "One-line summary"    # ≤1024 chars, shown to LLM

model: claude-sonnet-4-5-20250514  # Per-skill model override
context: fork                      # Run in isolated subagent
argument-hint: "<query>"           # Autocomplete hint for slash commands
user-invocable: true               # Enable /skill-name slash command

allowed-tools:                     # Restrict available tools
  - Read
  - Grep
  - Bash

hooks:                             # Lifecycle hooks
  PreToolExecution: "echo pre"
  PostToolExecution: "echo post"

metadata:
  emoji: "🔧"
  requires:
    bins: [git, gh]                # ALL must exist
    any_bins: [npm, pnpm]          # At least ONE
    env: [GITHUB_TOKEN]            # Required env vars
    os: [darwin, linux]            # Supported platforms
  primary_env: "API_KEY"           # Auto-inject for this skill
---
```

### Variable Substitution

| Placeholder | Description |
|-------------|-------------|
| `$ARGUMENTS` | Full arguments string |
| `$1`, `$2`, ... `$N` | Positional arguments |
| `${CLAUDE_SESSION_ID}` | Current session ID |
| `` !`command` `` | Replaced with stdout before sending to LLM |

## Example Skills

| Skill | Description | Deps |
|-------|-------------|------|
| **pdf** | PDF extraction, merging, splitting, form filling | pypdf, pdfplumber, reportlab |
| **pptx** | PowerPoint creation and editing | python-pptx, markitdown |
| **algorithmic-art** | Generative art with p5.js | p5.js, HTML/JS |
| **slack-gif-creator** | Animated GIF creation | PIL/Pillow |
| **web-artifacts-builder** | React + Tailwind + shadcn/ui apps | Node.js, Vite |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 AgentRunner                      │
│  - System prompt with skill discovery           │
│  - On-demand skill loading via tool call        │
│  - Slash commands + argument substitution       │
│  - Per-skill model switching + tool restriction │
│  - Context fork for isolated execution          │
├─────────────────────────────────────────────────┤
│                 SkillsEngine                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Loader  │  │ Filter  │  │ Runtime │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       ▼            ▼            ▼              │
│  ┌─────────────────────────────────────┐       │
│  │          SkillSnapshot              │       │
│  │  - Discovered skills + metadata     │       │
│  │  - System prompt fragment           │       │
│  └─────────────────────────────────────┘       │
├─────────────────────────────────────────────────┤
│          Event System (P0 roadmap)               │
│  before_tool_call · after_tool_result            │
│  context_transform · turn_start · turn_end       │
└─────────────────────────────────────────────────┘
        │              │              │
     OpenAI       Anthropic       MiniMax / Local
```

## Extending SkillEngine

### Custom Loader

```python
from skillengine.loaders import SkillLoader

class YAMLSkillLoader(SkillLoader):
    def can_load(self, path: Path) -> bool:
        return path.suffix == ".yaml"

    def load_skill(self, path: Path, source: SkillSource) -> SkillEntry:
        ...
```

### Custom Filter

```python
from skillengine.filters import SkillFilter

class TeamSkillFilter(SkillFilter):
    def filter(self, skill, config, context) -> FilterResult:
        if "team-only" in skill.metadata.tags:
            if not self.is_team_member():
                return FilterResult(skill, False, "Team members only")
        return FilterResult(skill, True)
```

### Custom Runtime

```python
from skillengine.runtime import SkillRuntime

class DockerRuntime(SkillRuntime):
    async def execute(self, command, cwd, env, timeout):
        # Execute in Docker container
        ...
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full technical roadmap. Summary:

| Phase | Priority | Feature |
|-------|----------|---------|
| 1 | P0 | Event system (lifecycle hooks, tool guards) |
| 1 | P0 | Structured stream events (thinking, tools, text) |
| 2 | P1 | Model metadata registry (cost, context window, capabilities) |
| 2 | P1 | Context management pipeline (compaction, pruning, token budgets) |
| 3 | P2 | Tool execution streaming (live output) |
| 3 | P2 | Steering & abort (cancel, interrupt, redirect mid-execution) |
| 4 | P3 | Dynamic provider registry (runtime adapter switching) |

## Development

```bash
git clone https://github.com/sawzhang/skillengine.git
cd skillengine
uv sync

# Run tests
pytest

# Linting
ruff check src/
ruff format src/
mypy src/
```

## License

MIT
