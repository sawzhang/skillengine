# Roadmap

SkillEngine has matured well past the original P0–P3 design targets in the previous roadmap (Event System, Structured Stream Events, Model Registry, Context Pipeline, Tool Streaming, Steering/Abort, Dynamic Provider Registry — **all shipped**). This document captures the **next four releases** that move SkillEngine from a feature-complete Alpha to a production-grade, ecosystem-integrated 1.0.

Reference scan date: 2026-05.

## Current Snapshot (what's already in `main`)

| Subsystem | Status | Module |
|---|---|---|
| Markdown+YAML skill DSL | ✅ Solid | `loaders/markdown.py` |
| Progressive disclosure (`skill` tool) | ✅ Solid | `agent.py` |
| Multi-provider adapters (OpenAI, Anthropic) | ✅ Solid | `adapters/` |
| Built-in tools (bash/read/write/edit/grep/find/ls/apply_patch/apply_diff) | ✅ Solid | `tools/` |
| Event bus (13 events, blockable) | ✅ Solid | `events.py` |
| Structured stream events | ✅ Solid | `adapters/*`, `agent.py` |
| Context compaction (token budget / sliding window) | ✅ Basic | `context.py` |
| Steering / Abort / Follow-up | ✅ Solid | `agent.py`, `modes/rpc_mode.py` |
| Sessions (JSONL append-only tree) | ✅ Solid | `session/` |
| Model registry & cost | ✅ Solid | `model_registry.py`, `models_catalog.py` |
| A2A (native protocol) | ✅ Basic | `a2a/` |
| Sandbox (BoxLite micro-VM + subprocess) | ✅ Basic | `sandbox/`, `runtime/boxlite.py` |
| Code-mode runtime (search+execute pattern) | ✅ Basic | `runtime/code_mode.py` |
| Memory (OpenViking REST backend) | ✅ Basic | `memory/` |
| Eval harness | 🟡 Skeleton | `harness/` |
| Optimizer (prompt improvement) | 🟡 Skeleton | `optimizer/` |
| Telemetry (OTel optional) | 🟡 Skeleton | `telemetry.py` |
| Auth / OAuth | 🟡 Skeleton | `auth/oauth.py` |
| Modes: interactive / json / rpc | ✅ Solid | `modes/` |
| TUI (renderer/editor/markdown/theme) | ✅ Solid | `tui/` |
| Web UI (SSE + SQLite) | 🟡 Minimal | `web/` |

## Gap Analysis vs. Industry Agent SDKs

Compared against Claude Agent SDK, OpenAI Agents SDK, LangGraph, Pydantic-AI, AG2/AutoGen, Mastra, and Vercel AI SDK:

**SkillEngine leads on**: Markdown skill DSL, progressive disclosure, BoxLite VM sandbox, JSONL session tree, hot-reload.

**SkillEngine is missing**:
1. **MCP interop** (every competitor supports MCP — we don't)
2. **Typed/structured output** (`output_type=PydanticModel`)
3. **Guardrails as a first-class abstraction** (we only have `BEFORE_TOOL_CALL` blocking)
4. **End-to-end tracing** with standardized span schema and cost attributes
5. **Eval/dataset/scorer abstraction** (harness skeleton only)
6. **Real summarizing compaction** (we only have prune/window)
7. **Workflow / DAG abstraction** (only `context: fork` and events)
8. **Computer-use / browser tools**
9. **Skill marketplace** (packages source exists, no version/signing/index)

The four releases below close these gaps in dependency order.

---

## v0.3 — Interop Release

**Theme**: make SkillEngine visible inside the wider ecosystem.

| ID | Item | Files |
|---|---|---|
| MCP-IN-1 | **MCP client** — expose remote MCP servers as local tools (stdio + SSE transport) | `src/skillengine/mcp/client.py` |
| MCP-IN-2 | **MCP server** — expose local Skills/Tools to Claude Desktop / Cursor / Cline | `src/skillengine/mcp/server.py` |
| MCP-IN-3 | **`mcp://` package source** — install MCP servers like skill packages | `src/skillengine/packages/source.py` |
| A2A-1 | **Protocol alignment** — OpenAI Handoffs + Anthropic A2A compat shims | `src/skillengine/a2a/handoffs.py` |
| TYPED-1 | **Typed output** — `agent.chat(..., output_type=PydanticModel)` with strict-JSON / tool-as-output strategy, streaming via `partial-json-parser` | `src/skillengine/agent.py`, `adapters/*` |
| DOC-1 | **Concept / cookbook / reference docs site** — 3-layer structure with 5 end-to-end cookbooks | `docs/` |

**Acceptance**: `npx @modelcontextprotocol/inspector stdio python -m skillengine.mcp.server` enumerates Skills as MCP tools. Cursor can call a local Skill via MCP. `agent.chat("...", output_type=MyModel)` returns a validated instance.

---

## v0.4 — Production Release

**Theme**: observability, validation, regression.

| ID | Item | Files |
|---|---|---|
| GUARD-1 | **Guardrails first-class** — `InputGuardrail` / `OutputGuardrail` / `ToolGuardrail`; built-ins: PII, prompt-injection, cost-budget, token-budget | `src/skillengine/guardrails/` |
| TRACE-1 | **End-to-end tracing** — span schema (`agent.turn`, `tool.call`, `skill.load`, `compact.run`), token/cost/cache/abort as attributes, exporters: console, OTel, LangSmith, Logfire | `src/skillengine/telemetry.py` |
| EVAL-1 | **Eval harness upgrade** — datasets, scorers (exact/contains/llm-judge/structured-match), `skills eval` CLI, 30+ built-in regression cases for the skill DSL | `src/skillengine/harness/`, `src/skillengine/cli.py` |
| CTX-1 | **Real compaction strategies** — `SummarizingCompactor` (per-segment summary), `ToolResultTruncator` (token-bounded), multi-modal token accounting | `src/skillengine/context.py` |
| COST-1 | **Cost dashboard** — `agent.cost_report()` per-skill / per-model / cache-hit-rate; visible in TUI and Web UI | `agent.py`, `tui/`, `web/` |
| TYPE-1 | **mypy strict-list cleanup** — remove `agent`, `runtime/*`, `session/*`, `memory/*`, `adapters/openai` from the relaxed override list | `pyproject.toml`, multiple |

**Acceptance**: `skills eval --suite skill-dsl` produces a leaderboard. TUI shows live token + cost + cache hit. Logfire/LangSmith shows a full trace tree. `mypy src/` passes strict for the listed modules.

---

## v0.5 — Capability Release

**Theme**: workflows, browser, marketplace.

| ID | Item | Files |
|---|---|---|
| FLOW-1 | ✅ **Workflow abstraction** — DAG nodes (agent / tool / branch / parallel / retry / checkpoint), serializable | `src/skillengine/workflow/` |
| FLOW-2 | ✅ **Durable execution** — checkpoint persistence to session tree, `--resume <session-id>` | `src/skillengine/workflow/`, `session/` |
| CUA-1 | **Computer-use & browser tools** — `browser_*` (Playwright), optional `computer_use` (Anthropic computer-use API) | `src/skillengine/tools/browser.py`, `tools/computer_use.py` |
| MARKET-1 | **Skill marketplace** — version constraints, signature verification, `skills install <name>@<ver>`, official index | `src/skillengine/packages/` |
| AUTH-1 | **Secret management** — keyring / sops / env-vars, per-skill secret injection, OAuth token refresh | `src/skillengine/auth/` |
| SCH-1 | **Scheduler upgrade** — durable task table, Web UI surface | `src/skillengine/scheduler.py`, `web/` |

**Acceptance**: a "Monday 9am, scrape GitHub issues → run `pdf` skill → publish report" workflow runs end-to-end with checkpointing.

---

## v1.0 — Stability Release

**Theme**: API freeze, performance, long-term support.

| ID | Item |
|---|---|
| STAB-1 | Public surface freeze + SemVer commitment + deprecation policy |
| PERF-1 | Benchmarks suite: cold-start, skill discovery, context compaction, stream latency; CI regression gates |
| CAT-1 | Model catalog automation — weekly auto-pull provider pricing, optional `skillengine-catalog` sub-package |
| A11Y-1 | Web UI reaches feature parity with TUI (commands, autocomplete, themes) |
| GA | Switch to `Development Status :: 5 - Production/Stable`; publish 0.x → 1.0 migration guide |

---

## Working Principles

- **No dead code in roadmap** — every item lands behind tests; ship in small slices.
- **Backwards compatibility from v0.3** — pre-1.0 still, but we stop breaking public agent APIs casually.
- **Each release closes one gap class** — interop, production, capability, stability.
- **Docs are a release blocker** — every release ships a concept page + cookbook.
