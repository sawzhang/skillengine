"""Built-in guardrail implementations.

Each guardrail is intentionally small and dependency-free. They are heuristic
defaults — production deployments should layer on something stronger (e.g. a
proper PII model or an LLM-judge for prompt-injection).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .base import Guardrail, GuardrailResult, GuardrailScope

__all__ = [
    "PIIGuardrail",
    "PromptInjectionGuardrail",
    "TokenBudgetGuardrail",
    "CostBudgetGuardrail",
]


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

_DEFAULT_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])?\d{3,4}[\s.-]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
}


@dataclass
class PIIGuardrail(Guardrail):
    """Detect personally identifiable information.

    By default, *redacts* matches by replacing them with ``[REDACTED:<kind>]``.
    Set ``mode="block"`` to refuse the payload instead.
    """

    name: str = "pii"
    scope: GuardrailScope = GuardrailScope.INPUT
    mode: str = "redact"  # "redact" or "block"
    patterns: dict[str, re.Pattern[str]] = field(
        default_factory=lambda: dict(_DEFAULT_PII_PATTERNS)
    )

    async def check(self, payload: Any, context: dict[str, Any]) -> GuardrailResult:
        text = _coerce_text(payload)
        if text is None:
            return GuardrailResult.allow()
        hits: dict[str, int] = {}
        redacted = text
        for kind, pattern in self.patterns.items():
            new_text, count = pattern.subn(f"[REDACTED:{kind}]", redacted)
            if count > 0:
                hits[kind] = count
                redacted = new_text
        if not hits:
            return GuardrailResult.allow()
        if self.mode == "block":
            return GuardrailResult.block(
                f"Input contains PII ({', '.join(hits)}); refusing.", hits=hits
            )
        # redact
        if isinstance(payload, dict):
            return GuardrailResult.transform(
                _set_first_string(payload, redacted),
                reason=f"Redacted PII: {hits}",
                hits=hits,
            )
        return GuardrailResult.transform(redacted, reason=f"Redacted PII: {hits}", hits=hits)


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

_DEFAULT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+|the\s+|previous\s+)+instructions?\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+|the\s+|previous\s+)+instructions?\b", re.I),
    re.compile(r"\boverride\s+(all\s+|the\s+|previous\s+)?(system|safety)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an)\s+\w+", re.I),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are)\b", re.I),
    re.compile(r"###\s*system\s*###", re.I),
    re.compile(r"```\s*system", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"<\s*\|im_start\s*\|>\s*system", re.I),
)


@dataclass
class PromptInjectionGuardrail(Guardrail):
    """Heuristic detector for prompt-injection payloads.

    Matches a curated set of suspicious patterns (e.g. *"ignore previous
    instructions"*). On a hit the default is to **block** with an explanatory
    message; set ``mode="redact"`` to scrub the offending lines instead.
    """

    name: str = "prompt_injection"
    scope: GuardrailScope = GuardrailScope.INPUT
    mode: str = "block"  # "block" or "redact"
    patterns: tuple[re.Pattern[str], ...] = _DEFAULT_INJECTION_PATTERNS

    async def check(self, payload: Any, context: dict[str, Any]) -> GuardrailResult:
        text = _coerce_text(payload)
        if text is None:
            return GuardrailResult.allow()
        matched: list[str] = [p.pattern for p in self.patterns if p.search(text)]
        if not matched:
            return GuardrailResult.allow()
        if self.mode == "redact":
            cleaned = text
            for p in self.patterns:
                cleaned = p.sub("[REDACTED:injection]", cleaned)
            replacement: Any = cleaned
            if isinstance(payload, dict):
                replacement = _set_first_string(payload, cleaned)
            return GuardrailResult.transform(
                replacement,
                reason=f"Removed suspected prompt-injection ({len(matched)} hits).",
                matched=matched,
            )
        return GuardrailResult.block(
            "Suspected prompt-injection in payload; refusing.", matched=matched
        )


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass
class TokenBudgetGuardrail(Guardrail):
    """Caps the total number of tokens consumed in a session.

    The guardrail tracks tokens via :meth:`record` (called by ``AgentRunner``
    after each LLM call) and blocks further INPUT once ``max_tokens`` is
    exceeded.
    """

    max_tokens: int = 100_000
    name: str = "token_budget"
    scope: GuardrailScope = GuardrailScope.INPUT
    used_tokens: int = 0

    def record(self, tokens: int) -> None:
        self.used_tokens += max(0, int(tokens))

    def reset(self) -> None:
        self.used_tokens = 0

    async def check(self, payload: Any, context: dict[str, Any]) -> GuardrailResult:
        if self.used_tokens >= self.max_tokens:
            return GuardrailResult.block(
                f"Token budget exceeded: {self.used_tokens}/{self.max_tokens}.",
                used=self.used_tokens,
                limit=self.max_tokens,
            )
        return GuardrailResult.allow()


@dataclass
class CostBudgetGuardrail(Guardrail):
    """Caps the total USD cost consumed in a session.

    The guardrail tracks cost via :meth:`record` (or :meth:`add_usage` if a
    :class:`~skillengine.model_registry.CostBreakdown` is available) and blocks
    further INPUT once ``max_cost_usd`` is exceeded.
    """

    max_cost_usd: float = 1.0
    name: str = "cost_budget"
    scope: GuardrailScope = GuardrailScope.INPUT
    used_cost: float = 0.0

    def record(self, cost_usd: float) -> None:
        self.used_cost += max(0.0, float(cost_usd))

    def add_usage(self, breakdown: Any) -> None:
        """Convenience: extract ``total_cost`` from a CostBreakdown-like value."""
        cost = getattr(breakdown, "total_cost", None)
        if cost is None and isinstance(breakdown, dict):
            cost = breakdown.get("total_cost")
        if cost is not None:
            self.record(float(cost))

    def reset(self) -> None:
        self.used_cost = 0.0

    async def check(self, payload: Any, context: dict[str, Any]) -> GuardrailResult:
        if self.used_cost >= self.max_cost_usd:
            return GuardrailResult.block(
                f"Cost budget exceeded: ${self.used_cost:.4f}/${self.max_cost_usd:.4f}.",
                used=self.used_cost,
                limit=self.max_cost_usd,
            )
        return GuardrailResult.allow()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_text(payload: Any) -> str | None:
    """Pull a string out of ``payload`` for text-based guardrails.

    For a dict (tool args), the first string-valued field is used. Returns
    ``None`` when nothing scan-worthy is found.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, str):
                return value
    return None


def _set_first_string(payload: dict[str, Any], replacement: str) -> dict[str, Any]:
    """Return a copy of ``payload`` with its first string-valued field replaced."""
    out = dict(payload)
    for key, value in payload.items():
        if isinstance(value, str):
            out[key] = replacement
            break
    return out
