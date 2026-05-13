"""Guardrails for SkillEngine — input, output, and tool-call validation.

A guardrail inspects a piece of agent activity (user input, model output, or a
proposed tool call) and decides whether to *allow*, *block*, or *transform* it.
Guardrails are framework-agnostic and event-driven: a :class:`GuardrailManager`
binds them to the agent's :class:`~skillengine.events.EventBus`.

Built-in guardrails:
    * :class:`PIIGuardrail` — redacts or blocks personally identifiable info.
    * :class:`PromptInjectionGuardrail` — heuristic detector for prompt-injection
      payloads in user input or tool output.
    * :class:`TokenBudgetGuardrail` — caps total tokens spent in a chat session.
    * :class:`CostBudgetGuardrail` — caps total cost (USD) spent in a session.
"""

from .base import (
    Guardrail,
    GuardrailAction,
    GuardrailManager,
    GuardrailResult,
    GuardrailScope,
    GuardrailViolation,
)
from .builtins import (
    CostBudgetGuardrail,
    PIIGuardrail,
    PromptInjectionGuardrail,
    TokenBudgetGuardrail,
)

__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailManager",
    "GuardrailResult",
    "GuardrailScope",
    "GuardrailViolation",
    "PIIGuardrail",
    "PromptInjectionGuardrail",
    "TokenBudgetGuardrail",
    "CostBudgetGuardrail",
]
