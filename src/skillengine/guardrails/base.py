"""Guardrail abstractions and the manager that binds them to the event bus."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from skillengine.events import (
    AFTER_TOOL_RESULT,
    BEFORE_TOOL_CALL,
    INPUT,
    AfterToolResultEvent,
    BeforeToolCallEvent,
    EventBus,
    InputEvent,
    InputEventResult,
    ToolCallEventResult,
    ToolResultEventResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailManager",
    "GuardrailResult",
    "GuardrailScope",
    "GuardrailViolation",
]


class GuardrailScope(str, Enum):
    """Where a guardrail applies."""

    INPUT = "input"
    OUTPUT = "output"
    TOOL = "tool"


class GuardrailAction(str, Enum):
    """What a guardrail wants to do with the payload."""

    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


@dataclass
class GuardrailResult:
    """Result of a single guardrail check.

    Attributes:
        action: ``ALLOW`` / ``BLOCK`` / ``TRANSFORM``.
        reason: Human-readable reason (always populated for non-allow actions).
        replacement: For ``TRANSFORM``, the new payload (string for input/output,
            ``dict`` for tool args).
        metadata: Optional extra info for logging / telemetry.
    """

    action: GuardrailAction = GuardrailAction.ALLOW
    reason: str = ""
    replacement: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> GuardrailResult:
        return cls(action=GuardrailAction.ALLOW)

    @classmethod
    def block(cls, reason: str, **metadata: Any) -> GuardrailResult:
        return cls(action=GuardrailAction.BLOCK, reason=reason, metadata=dict(metadata))

    @classmethod
    def transform(cls, replacement: Any, reason: str = "", **metadata: Any) -> GuardrailResult:
        return cls(
            action=GuardrailAction.TRANSFORM,
            reason=reason,
            replacement=replacement,
            metadata=dict(metadata),
        )


class GuardrailViolation(RuntimeError):  # noqa: N818 - "Error" suffix less natural here
    """Raised when a guardrail blocks an operation and the caller asked to raise."""

    def __init__(self, reason: str, *, scope: GuardrailScope, name: str) -> None:
        super().__init__(f"[{scope.value}/{name}] {reason}")
        self.reason = reason
        self.scope = scope
        self.name = name


class Guardrail(ABC):
    """Base class for guardrails.

    Override :meth:`check` (sync or async). The ``scope`` attribute determines
    which event the manager binds the guardrail to.
    """

    name: str = "guardrail"
    scope: GuardrailScope = GuardrailScope.INPUT

    @abstractmethod
    async def check(self, payload: Any, context: dict[str, Any]) -> GuardrailResult:
        """Inspect ``payload`` and return a :class:`GuardrailResult`."""


class GuardrailManager:
    """Binds a collection of :class:`Guardrail` instances to an :class:`EventBus`.

    Usage::

        manager = GuardrailManager(event_bus)
        manager.add(PIIGuardrail())
        manager.add(TokenBudgetGuardrail(max_tokens=10_000))

    The manager subscribes the appropriate event handlers automatically. Call
    :meth:`detach` to unsubscribe.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.bus = event_bus
        self._guardrails: list[Guardrail] = []
        self._attached = False
        self._unsubscribers: list[Any] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, guardrail: Guardrail) -> None:
        self._guardrails.append(guardrail)
        if not self._attached:
            self._attach()

    def extend(self, guardrails: list[Guardrail]) -> None:
        for g in guardrails:
            self.add(g)

    def all(self) -> list[Guardrail]:
        return list(self._guardrails)

    def clear(self) -> None:
        self._guardrails.clear()

    def by_scope(self, scope: GuardrailScope) -> list[Guardrail]:
        return [g for g in self._guardrails if g.scope == scope]

    # ------------------------------------------------------------------
    # Direct invocation (for OUTPUT scope and explicit calls)
    # ------------------------------------------------------------------

    async def check_output(
        self,
        text: str,
        *,
        context: dict[str, Any] | None = None,
        raise_on_block: bool = False,
    ) -> tuple[str, list[GuardrailResult]]:
        """Run every OUTPUT guardrail against ``text``.

        Returns ``(final_text, results)``. Transformations are applied in order;
        a block short-circuits and (optionally) raises.
        """
        results: list[GuardrailResult] = []
        ctx = dict(context or {})
        current = text
        for g in self.by_scope(GuardrailScope.OUTPUT):
            result = await _maybe_await(g.check(current, ctx))
            results.append(result)
            if result.action == GuardrailAction.BLOCK:
                if raise_on_block:
                    raise GuardrailViolation(
                        result.reason, scope=GuardrailScope.OUTPUT, name=g.name
                    )
                logger.info("Output guardrail '%s' blocked: %s", g.name, result.reason)
                return result.reason or "[blocked by output guardrail]", results
            if result.action == GuardrailAction.TRANSFORM and isinstance(result.replacement, str):
                current = result.replacement
        return current, results

    # ------------------------------------------------------------------
    # Internal: event binding
    # ------------------------------------------------------------------

    def _attach(self) -> None:
        self._unsubscribers = [
            self.bus.on(INPUT, self._on_input, priority=100),
            self.bus.on(BEFORE_TOOL_CALL, self._on_before_tool_call, priority=100),
            self.bus.on(AFTER_TOOL_RESULT, self._on_after_tool_result, priority=100),
        ]
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception:  # pragma: no cover - defensive
                pass
        self._unsubscribers.clear()
        self._attached = False

    async def _on_input(self, event: InputEvent) -> InputEventResult | None:
        text = event.user_input
        ctx: dict[str, Any] = {"event": event}
        for g in self.by_scope(GuardrailScope.INPUT):
            result = await _maybe_await(g.check(text, ctx))
            if result.action == GuardrailAction.BLOCK:
                logger.info("Input guardrail '%s' blocked: %s", g.name, result.reason)
                return InputEventResult(
                    action="handled",
                    response=result.reason or "[blocked by input guardrail]",
                )
            if result.action == GuardrailAction.TRANSFORM and isinstance(result.replacement, str):
                text = result.replacement
        if text != event.user_input:
            return InputEventResult(action="transform", transformed_input=text)
        return None

    async def _on_before_tool_call(self, event: BeforeToolCallEvent) -> ToolCallEventResult | None:
        args = dict(event.args)
        ctx: dict[str, Any] = {
            "event": event,
            "tool_name": event.tool_name,
            "tool_call_id": event.tool_call_id,
        }
        changed = False
        for g in self.by_scope(GuardrailScope.TOOL):
            result = await _maybe_await(g.check(args, ctx))
            if result.action == GuardrailAction.BLOCK:
                logger.info(
                    "Tool guardrail '%s' blocked %s: %s",
                    g.name,
                    event.tool_name,
                    result.reason,
                )
                return ToolCallEventResult(block=True, reason=result.reason)
            if result.action == GuardrailAction.TRANSFORM and isinstance(result.replacement, dict):
                args = result.replacement
                changed = True
        if changed:
            return ToolCallEventResult(modified_args=args)
        return None

    async def _on_after_tool_result(
        self, event: AfterToolResultEvent
    ) -> ToolResultEventResult | None:
        # Run OUTPUT-scope guardrails on tool results too (defence-in-depth
        # against prompt injection delivered via tool stdout).
        text = event.result
        ctx: dict[str, Any] = {
            "event": event,
            "source": "tool_result",
            "tool_name": event.tool_name,
        }
        original = text
        for g in self.by_scope(GuardrailScope.OUTPUT):
            result = await _maybe_await(g.check(text, ctx))
            if result.action == GuardrailAction.BLOCK:
                return ToolResultEventResult(
                    modified_result=result.reason or "[blocked by output guardrail]"
                )
            if result.action == GuardrailAction.TRANSFORM and isinstance(result.replacement, str):
                text = result.replacement
        if text != original:
            return ToolResultEventResult(modified_result=text)
        return None


async def _maybe_await(value: Any) -> GuardrailResult:
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, GuardrailResult):  # pragma: no cover - defensive
        raise TypeError(f"Guardrail.check must return GuardrailResult, got {type(value)!r}")
    return value
