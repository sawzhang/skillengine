# 02 — Add a local skill

Goal: teach the agent a new capability without writing Python.

## 1. Create the skill file

```
mkdir -p skills/greet
```

`skills/greet/SKILL.md`:

```markdown
---
name: greet
description: Greet the user warmly by name, in their preferred language.
argument-hint: "<name> [language]"
---

Greet $1 warmly in ${2:-English}. Keep it to one sentence.
```

## 2. Point the engine at the directory

```python
engine = SkillsEngine(SkillsConfig(skill_dirs=["./skills"]))
runner = AgentRunner(engine=engine, config=config)
print(await runner.chat("Use the greet skill for Sora in Japanese."))
```

## 3. What the LLM sees

System prompt (built by `AgentRunner`):

```
You have access to the following skills:
- greet: Greet the user warmly by name, in their preferred language.

When you want to use a skill call the `skill` tool with its name.
```

The full body of `greet` is **not** in the prompt — it is delivered on demand
when the LLM calls `skill(name="greet", arguments="Sora Japanese")`. This is the
*progressive disclosure* pattern; see
[`../concepts/skills.md`](../concepts/skills.md).

## Going further

- Add `requires.bins: [git]` to gate a skill on a binary being installed.
- Add `model: claude-sonnet-4-20250514` to run a single skill on a smarter model.
- Add `context: fork` to isolate the skill's reasoning from the parent.
