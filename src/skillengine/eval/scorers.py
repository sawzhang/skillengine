"""Built-in scorers for the eval harness."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Scorer",
    "ScorerResult",
    "ExactMatchScorer",
    "ContainsScorer",
    "RegexScorer",
    "StructuredMatchScorer",
    "LLMJudgeScorer",
]


@dataclass
class ScorerResult:
    """Result of evaluating one case with one scorer."""

    name: str
    passed: bool
    score: float  # in [0.0, 1.0]
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    """Abstract scorer.

    Override :meth:`score`. ``case_expected`` is whatever the dataset's
    ``EvalCase.expected`` field contains.
    """

    name: str = "scorer"

    @abstractmethod
    async def score(self, actual: Any, case_expected: Any) -> ScorerResult: ...


# ---------------------------------------------------------------------------
# Text-based
# ---------------------------------------------------------------------------


@dataclass
class ExactMatchScorer(Scorer):
    """Pass iff ``str(actual) == str(expected)``.

    Set ``case_insensitive=True`` to ignore case, ``strip=True`` to ignore
    surrounding whitespace.
    """

    case_insensitive: bool = False
    strip: bool = True
    name: str = "exact_match"

    async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
        a, b = str(actual), str(case_expected)
        if self.strip:
            a, b = a.strip(), b.strip()
        if self.case_insensitive:
            a, b = a.lower(), b.lower()
        ok = a == b
        return ScorerResult(
            name=self.name,
            passed=ok,
            score=1.0 if ok else 0.0,
            reason="" if ok else f"actual!=expected ({a!r} vs {b!r})",
        )


@dataclass
class ContainsScorer(Scorer):
    """Pass iff ``actual`` contains ``expected`` (substring or all of a list).

    If ``expected`` is a list, every element must appear in ``actual``.
    """

    case_insensitive: bool = True
    name: str = "contains"

    async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
        text = str(actual)
        needles: list[str]
        if isinstance(case_expected, str):
            needles = [case_expected]
        elif isinstance(case_expected, (list, tuple)):
            needles = [str(x) for x in case_expected]
        else:
            needles = [str(case_expected)]
        if self.case_insensitive:
            text_l = text.lower()
            hits = [n for n in needles if n.lower() in text_l]
        else:
            hits = [n for n in needles if n in text]
        missing = [n for n in needles if n not in hits]
        score = len(hits) / len(needles) if needles else 0.0
        return ScorerResult(
            name=self.name,
            passed=not missing,
            score=score,
            reason="" if not missing else f"missing: {missing}",
            details={"hits": hits, "missing": missing},
        )


@dataclass
class RegexScorer(Scorer):
    """Pass iff ``actual`` matches the regex in ``expected``."""

    flags: int = 0
    fullmatch: bool = False
    name: str = "regex"

    async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
        pattern = re.compile(str(case_expected), self.flags)
        text = str(actual)
        m = pattern.fullmatch(text) if self.fullmatch else pattern.search(text)
        return ScorerResult(
            name=self.name,
            passed=m is not None,
            score=1.0 if m else 0.0,
            reason="" if m else f"no match for {case_expected!r}",
            details={"match": m.group(0) if m else None},
        )


# ---------------------------------------------------------------------------
# Structured
# ---------------------------------------------------------------------------


@dataclass
class StructuredMatchScorer(Scorer):
    """Pass iff every key in ``expected`` is present in ``actual`` with the
    same value.

    Both must be JSON-like (``dict``, ``list``, primitives). If ``actual`` is a
    JSON string, it is parsed. Extra keys in ``actual`` are allowed unless
    ``strict=True``.
    """

    strict: bool = False
    name: str = "structured_match"

    async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
        import json

        parsed = actual
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                return ScorerResult(
                    name=self.name,
                    passed=False,
                    score=0.0,
                    reason="actual is not valid JSON",
                )
        ok, diff = _deep_subset(parsed, case_expected, strict=self.strict)
        return ScorerResult(
            name=self.name,
            passed=ok,
            score=1.0 if ok else 0.0,
            reason="" if ok else f"mismatch: {diff}",
            details={"diff": diff},
        )


def _deep_subset(actual: Any, expected: Any, *, strict: bool) -> tuple[bool, str]:
    """Recursive subset comparison. Returns (ok, reason)."""
    if type(actual) is not type(expected) and not (
        isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    ):
        return False, f"type mismatch: {type(actual).__name__} vs {type(expected).__name__}"
    if isinstance(expected, dict):
        if strict and set(actual.keys()) != set(expected.keys()):
            extra = set(actual.keys()) - set(expected.keys())
            missing = set(expected.keys()) - set(actual.keys())
            return False, f"keys differ (extra={extra}, missing={missing})"
        for k, v in expected.items():
            if k not in actual:
                return False, f"missing key {k!r}"
            ok, why = _deep_subset(actual[k], v, strict=strict)
            if not ok:
                return False, f"{k}: {why}"
        return True, ""
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return False, f"list length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            ok, why = _deep_subset(a, e, strict=strict)
            if not ok:
                return False, f"[{i}]: {why}"
        return True, ""
    if actual != expected:
        return False, f"{actual!r} != {expected!r}"
    return True, ""


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeScorer(Scorer):
    """Ask an LLM whether ``actual`` matches ``expected``.

    ``judge`` is any async callable ``(prompt) -> str`` (e.g. an
    :class:`AgentRunner.chat` bound method or a thin wrapper around an
    adapter). The reply is expected to be ``PASS`` or ``FAIL`` followed by an
    optional reason.

    Strict parsing is intentional — the prompt makes the format obvious — but
    free-form replies that *start* with ``PASS``/``FAIL`` are also accepted.
    """

    judge: Any = None  # async callable(prompt) -> str
    criterion: str = "The actual answer is correct and addresses the expected behaviour."
    name: str = "llm_judge"
    prompt_template: str = (
        "You are a strict grader. Decide whether the model's actual reply "
        "satisfies the criterion below.\n\n"
        "Criterion: {criterion}\n\n"
        "Expected (reference, may be partial): {expected}\n\n"
        "Actual reply: {actual}\n\n"
        "Reply with exactly one line starting with PASS or FAIL, optionally "
        "followed by a one-sentence reason."
    )

    async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
        if self.judge is None:
            return ScorerResult(
                name=self.name, passed=False, score=0.0, reason="no judge configured"
            )
        prompt = self.prompt_template.format(
            criterion=self.criterion,
            expected=case_expected,
            actual=actual,
        )
        result = self.judge(prompt)
        if inspect.isawaitable(result):
            result = await result
        text = str(result).strip()
        first_token = text.split()[0].upper() if text else ""
        passed = first_token == "PASS"
        return ScorerResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=text,
            details={"raw": text},
        )
