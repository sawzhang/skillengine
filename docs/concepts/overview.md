# Overview

SkillEngine is a framework-agnostic execution engine for LLM agent **skills** —
units of capability defined as Markdown files with YAML frontmatter.

## The pipeline

```
Skill files (Markdown + YAML)
        │
        ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Loader  │───▶│ Filter  │───▶│ Runtime │───▶│ Adapter │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
   parse YAML    eligibility    shell / VM     LLM tool calls
                 (bins, env)    / code mode
```

- **Loader** — turns a `SKILL.md` into a `Skill` dataclass.
- **Filter** — decides whether the skill is eligible right now (required binaries,
  env vars, OS, custom predicates).
- **Runtime** — executes commands from skill bodies (bash, code-mode, sandboxed).
- **Adapter** — exposes skills as tools for OpenAI / Anthropic / OpenAI-compatible
  LLMs and translates tool-call results back.

## Why progressive disclosure

The system prompt contains **only skill names and descriptions**. The full body of
a skill is only loaded when the LLM calls the `skill` tool to ask for it. This is
the same pattern Claude Agent Skills uses, and it keeps the context window cheap
even with hundreds of skills installed.

See [`skills.md`](./skills.md) for the file format and
[`agents.md`](./agents.md) for how `AgentRunner` orchestrates this.
