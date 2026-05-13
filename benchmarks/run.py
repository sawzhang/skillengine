"""Benchmark runner for skillengine hot paths.

Run with::

    uv run python -m benchmarks.run               # print results
    uv run python -m benchmarks.run --save-baseline
    uv run python -m benchmarks.run --check       # compare vs baseline

This is intentionally dependency-free (only stdlib + skillengine itself).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skillengine.agent import AgentMessage
from skillengine.context import (
    TokenBudgetCompactor,
    estimate_messages_tokens,
)
from skillengine.events import AGENT_START, AgentStartEvent, EventBus
from skillengine.loaders.markdown import MarkdownSkillLoader
from skillengine.models import SkillSource
from skillengine.tools.dispatcher import ToolContext, ToolDispatcher

BASELINE_PATH = Path(__file__).parent / "baseline.json"
REGRESSION_THRESHOLD = 1.25  # >25% slower than baseline counts as a regression
WARMUP_TRIALS = 3
MEASURE_TRIALS = 30


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    median_ms: float
    p95_ms: float
    trials: int
    samples_ms: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.name:<28s} median={self.median_ms:7.3f}ms  p95={self.p95_ms:7.3f}ms"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_sync(fn: Callable[[], Any], trials: int) -> list[float]:
    samples: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


async def _time_async(fn: Callable[[], Awaitable[Any]], trials: int) -> list[float]:
    samples: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        await fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _summarize(name: str, samples: list[float]) -> BenchResult:
    samples_sorted = sorted(samples)
    median = statistics.median(samples_sorted)
    p95_index = max(0, int(len(samples_sorted) * 0.95) - 1)
    p95 = samples_sorted[p95_index]
    return BenchResult(name=name, median_ms=median, p95_ms=p95, trials=len(samples))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_messages(n: int) -> list[AgentMessage]:
    """Build a synthetic conversation with mixed roles."""
    msgs: list[AgentMessage] = []
    for i in range(n):
        role = ("user", "assistant", "tool")[i % 3]
        content = f"message {i}: " + ("lorem ipsum dolor sit amet " * 12)
        msgs.append(AgentMessage(role=role, content=content))
    return msgs


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def bench_token_estimation() -> BenchResult:
    msgs = _build_messages(1000)
    # warmup
    _time_sync(lambda: estimate_messages_tokens(msgs), WARMUP_TRIALS)
    samples = _time_sync(lambda: estimate_messages_tokens(msgs), MEASURE_TRIALS)
    return _summarize("token_estimation/1k_msgs", samples)


async def bench_context_compaction() -> BenchResult:
    msgs = _build_messages(1000)
    compactor = TokenBudgetCompactor()
    budget = 4000  # force aggressive compaction

    async def run() -> None:
        await compactor.compact(msgs, budget)

    await _time_async(run, WARMUP_TRIALS)
    samples = await _time_async(run, MEASURE_TRIALS)
    return _summarize("context_compaction/1k_msgs", samples)


async def bench_tool_dispatch() -> BenchResult:
    dispatcher = ToolDispatcher()

    async def noop_handler(
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
        on_output: Any,
    ) -> str:
        return "ok"

    dispatcher.register(
        "noop",
        noop_handler,
        {"type": "function", "function": {"name": "noop", "description": "noop"}},
    )
    ctx = ToolContext(
        engine=None,
        config=None,
        abort_event=asyncio.Event(),
        event_bus=EventBus(),
    )

    async def run() -> None:
        await dispatcher.dispatch("noop", {}, ctx)

    await _time_async(run, WARMUP_TRIALS * 100)
    samples = await _time_async(run, MEASURE_TRIALS * 100)
    # report per-call median
    return _summarize("tool_dispatch/single_call", samples)


async def bench_event_emit() -> BenchResult:
    bus = EventBus()

    def sync_handler(event: AgentStartEvent) -> None:
        return None

    async def async_handler(event: AgentStartEvent) -> None:
        return None

    for _ in range(4):
        bus.on(AGENT_START, sync_handler)
        bus.on(AGENT_START, async_handler)

    payload = AgentStartEvent(user_input="hello", system_prompt="sys", model="m")

    async def run() -> None:
        await bus.emit(AGENT_START, payload)

    await _time_async(run, WARMUP_TRIALS * 100)
    samples = await _time_async(run, MEASURE_TRIALS * 100)
    return _summarize("event_emit/8_handlers", samples)


def bench_skill_load() -> BenchResult:
    skills_dir = Path(__file__).parent.parent / "skills"
    if not skills_dir.exists():
        return BenchResult(
            name="skill_load/bundled (skipped — no skills/ dir)",
            median_ms=0.0,
            p95_ms=0.0,
            trials=0,
        )
    loader = MarkdownSkillLoader()

    def run() -> None:
        loader.load_directory(skills_dir, SkillSource.BUNDLED)

    _time_sync(run, WARMUP_TRIALS)
    samples = _time_sync(run, MEASURE_TRIALS)
    return _summarize("skill_load/bundled_dir", samples)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_all() -> list[BenchResult]:
    results: list[BenchResult] = []
    results.append(bench_token_estimation())
    results.append(await bench_context_compaction())
    results.append(await bench_tool_dispatch())
    results.append(await bench_event_emit())
    results.append(bench_skill_load())
    return results


def _save_baseline(results: list[BenchResult]) -> None:
    payload = {r.name: {"median_ms": r.median_ms, "p95_ms": r.p95_ms} for r in results}
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved baseline -> {BASELINE_PATH}")


def _check_baseline(results: list[BenchResult]) -> int:
    if not BASELINE_PATH.exists():
        print(
            f"No baseline at {BASELINE_PATH}; run --save-baseline first.",
            file=sys.stderr,
        )
        return 2
    baseline = json.loads(BASELINE_PATH.read_text())
    regressions: list[tuple[str, float, float]] = []
    for r in results:
        if r.trials == 0:
            continue
        prior = baseline.get(r.name)
        if prior is None:
            continue
        prior_med = float(prior.get("median_ms") or 0.0)
        if prior_med <= 0:
            continue
        ratio = r.median_ms / prior_med
        if ratio > REGRESSION_THRESHOLD:
            regressions.append((r.name, prior_med, r.median_ms))
    if regressions:
        print("\nREGRESSIONS DETECTED:", file=sys.stderr)
        for name, prior_med, now_med in regressions:
            pct = (now_med / prior_med - 1.0) * 100.0
            print(
                f"  {name}: {prior_med:.3f}ms -> {now_med:.3f}ms (+{pct:.1f}%)",
                file=sys.stderr,
            )
        return 1
    print("\nNo regressions vs. baseline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.run")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Emit results.json next to baseline.json"
    )
    args = parser.parse_args(argv)

    print("Running skillengine benchmarks...")
    results = asyncio.run(run_all())
    print()
    for r in results:
        if r.trials == 0:
            print(f"{r.name}  (skipped)")
        else:
            print(r.summary())

    if args.json:
        out = Path(__file__).parent / "results.json"
        out.write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n",
        )
        print(f"\nWrote {out}")

    if args.save_baseline:
        _save_baseline(results)
        return 0

    if args.check:
        return _check_baseline(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
