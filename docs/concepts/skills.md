# Skills

A skill is a directory with a `SKILL.md` file:

```
skills/
  my-skill/
    SKILL.md
    templates/      # optional helper assets
```

## File format

```markdown
---
name: my-skill
description: One-line description shown to the LLM.
model: claude-sonnet-4-20250514   # optional per-skill model
context: fork                      # optional isolated subagent
allowed-tools: [Read, Grep]        # optional tool restriction
argument-hint: "<query>"           # autocomplete hint
user-invocable: true               # enables /my-skill slash command
disable-model-invocation: false    # hide from LLM auto-discovery
metadata:
  requires:
    bins: [git]                    # ALL must be on PATH
    any_bins: [npm, pnpm]          # ONE must be on PATH
    env: [GITHUB_TOKEN]            # ALL must be set
    os: [darwin, linux]
---

# Skill body
The text below the frontmatter is the prompt the LLM sees when it loads
this skill. Use `$ARGUMENTS` (or `$1`, `$2`, ...) for dynamic input.

Current date: !`date +%Y-%m-%d`
```

## Substitution rules

| Token | Replaced with |
|---|---|
| `$ARGUMENTS` | the full argument string passed to the skill |
| `$1`, `$2`, ... | individual whitespace-separated arguments |
| `${CLAUDE_SESSION_ID}` | the current agent session id |
| `` !`command` `` | stdout of `command` (executed at load time) |

## Per-skill behaviour

- **`model:`** — calls `switch_model()` before running the skill body and restores
  the previous model in `finally`.
- **`context: fork`** — runs the skill body inside a fresh child `AgentRunner` with
  the body as its system prompt. The parent only sees the final answer.
- **`allowed-tools:`** — restricts which tools the LLM may call while this skill is
  active.

## Validation

`AgentRunner.validate_skill()` enforces:

- `name` ≤ 64 chars, `[a-z0-9-]+`, no leading hyphen
- `description` non-empty and ≤ 1024 chars
