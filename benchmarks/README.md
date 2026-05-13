# Benchmarks

Lightweight micro-benchmarks that establish a performance baseline for the
hot paths in `skillengine`. The goal is **regression detection**, not absolute
throughput tuning — every result is reported as a median + p95 over N trials
and compared against the previously-saved baseline in `baseline.json`.

## What is measured

| Suite | Function | Why it matters |
|---|---|---|
| `token_estimation` | `estimate_messages_tokens` over 1K-msg synthetic transcript | Called on every LLM turn for compaction decisions |
| `context_compaction` | `TokenBudgetCompactor.compact` on a 1K-msg transcript | Runs whenever the threshold is crossed; O(n) drop-from-front |
| `tool_dispatch` | `ToolDispatcher.dispatch` for a no-op handler | Once per tool call, on the hot path |
| `event_emit` | `EventBus.emit` to 8 handlers (4 sync + 4 async) | Fires up to 13× per turn (lifecycle + tool hooks) |
| `skill_load` | `MarkdownSkillLoader.load_directory` over the bundled `skills/` | One-time at startup, but watched/reloaded in dev |

## Running

```bash
# Run the suite (no external services required)
uv run python -m benchmarks.run

# Save current run as the new baseline (do this after intentional perf changes)
uv run python -m benchmarks.run --save-baseline

# Fail with non-zero exit code if any metric regresses > 25% vs. baseline
uv run python -m benchmarks.run --check
```

## CI integration

The suite is **not** wired into PR CI yet — it is too noisy on shared GitHub
runners. Recommended next step: add a nightly workflow on a self-hosted runner
that calls `--check` and uploads `results.json` as an artifact.
