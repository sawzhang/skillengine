"""EvalRunner: runs a target over a dataset and aggregates :class:`ScorerResult`s."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset import EvalCase, EvalDataset
from .scorers import Scorer, ScorerResult

logger = logging.getLogger(__name__)

__all__ = ["EvalRunner", "EvalCaseResult", "EvalReport"]


# Target callable: takes the case's input, returns the actual output.
TargetFn = Callable[[Any], Any]  # may be sync or async


@dataclass
class EvalCaseResult:
    """Per-case result with per-scorer breakdown."""

    case_id: str
    actual: Any
    scores: list[ScorerResult] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)

    @property
    def score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


@dataclass
class EvalReport:
    """Aggregate report over a dataset."""

    dataset: str
    cases: list[EvalCaseResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.cases if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def errored(self) -> int:
        return sum(1 for r in self.cases if r.error is not None)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(r.score for r in self.cases) / len(self.cases)

    def summary(self) -> str:
        return (
            f"{self.dataset}: {self.passed}/{self.total} passed "
            f"({self.pass_rate * 100:.1f}%), mean score {self.mean_score:.3f}, "
            f"{self.errored} errored, {self.duration_ms:.0f}ms total"
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        # asdict converts dataclasses recursively; add computed fields.
        out["total"] = self.total
        out["passed"] = self.passed
        out["failed"] = self.failed
        out["errored"] = self.errored
        out["pass_rate"] = self.pass_rate
        out["mean_score"] = self.mean_score
        return out

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Runs a target callable over an :class:`EvalDataset` with a list of scorers."""

    def __init__(
        self,
        target: TargetFn,
        scorers: list[Scorer],
        *,
        concurrency: int = 1,
        on_case: Callable[[EvalCaseResult], None] | None = None,
    ) -> None:
        if not scorers:
            raise ValueError("EvalRunner requires at least one scorer")
        self.target = target
        self.scorers = scorers
        self.concurrency = max(1, int(concurrency))
        self.on_case = on_case

    async def run(self, dataset: EvalDataset) -> EvalReport:
        start = time.time()
        sem = asyncio.Semaphore(self.concurrency)

        async def _run_case(case: EvalCase) -> EvalCaseResult:
            async with sem:
                return await self._run_one(case)

        coros = [_run_case(c) for c in dataset.cases]
        results = await asyncio.gather(*coros)
        report = EvalReport(
            dataset=dataset.name,
            cases=list(results),
            duration_ms=(time.time() - start) * 1000.0,
        )
        return report

    async def _run_one(self, case: EvalCase) -> EvalCaseResult:
        case_start = time.time()
        actual: Any = None
        error: str | None = None
        try:
            value = self.target(case.input)
            if inspect.isawaitable(value):
                value = await value
            actual = value
        except Exception as exc:  # noqa: BLE001 - exposed in report
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Target raised on case %s", case.id)

        scores: list[ScorerResult] = []
        if error is None:
            for scorer in self.scorers:
                try:
                    s = await scorer.score(actual, case.expected)
                except Exception as exc:  # noqa: BLE001
                    s = ScorerResult(
                        name=getattr(scorer, "name", type(scorer).__name__),
                        passed=False,
                        score=0.0,
                        reason=f"scorer error: {type(exc).__name__}: {exc}",
                    )
                scores.append(s)

        result = EvalCaseResult(
            case_id=case.id,
            actual=actual,
            scores=scores,
            duration_ms=(time.time() - case_start) * 1000.0,
            error=error,
        )
        if self.on_case is not None:
            try:
                self.on_case(result)
            except Exception:  # pragma: no cover - defensive
                logger.exception("on_case callback raised")
        return result


def _ensure_async(fn: TargetFn) -> Callable[[Any], Awaitable[Any]]:
    """Wrap a sync target so it can be awaited."""
    if inspect.iscoroutinefunction(fn):
        return fn  # type: ignore[return-value]

    async def _async(x: Any) -> Any:
        result = fn(x)
        if inspect.isawaitable(result):
            return await result
        return result

    return _async
