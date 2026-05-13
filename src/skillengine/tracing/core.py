"""Core tracing primitives: ``Span``, ``Tracer``, and event-bus wiring."""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from skillengine.events import (
    AFTER_TOOL_RESULT,
    AGENT_END,
    AGENT_START,
    BEFORE_TOOL_CALL,
    COMPACTION,
    MODEL_CHANGE,
    TURN_END,
    TURN_START,
)

if TYPE_CHECKING:
    from skillengine.events import EventBus

logger = logging.getLogger(__name__)

__all__ = [
    "Span",
    "SpanContext",
    "SpanExporter",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "install_tracer",
]


# ---------------------------------------------------------------------------
# Span data model
# ---------------------------------------------------------------------------


class SpanKind(str, Enum):
    AGENT = "agent"
    TURN = "turn"
    TOOL = "tool"
    SKILL = "skill"
    COMPACT = "compact"
    INTERNAL = "internal"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class SpanContext:
    """Identity of a trace/span. Cheap to copy and pass around."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None


@dataclass
class Span:
    """A single tracing span.

    Spans are *immutable IDs* with *mutable state* (attributes, events,
    end time). They are exported once :meth:`end` is called.
    """

    name: str
    kind: SpanKind
    context: SpanContext
    start_time: float
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    _tracer: Tracer | None = None
    _ended: bool = False

    # ----- attribute helpers --------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def set_attributes(self, **values: Any) -> Span:
        self.attributes.update(values)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": dict(attributes or {}),
            }
        )
        return self

    def record_exception(self, exc: BaseException) -> Span:
        self.status = SpanStatus.ERROR
        self.attributes["error"] = True
        self.attributes["error.message"] = str(exc)
        self.attributes["error.type"] = type(exc).__name__
        return self

    def end(self, *, status: SpanStatus | None = None) -> Span:
        if self._ended:
            return self
        self._ended = True
        self.end_time = time.time()
        if status is not None:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
        if self._tracer is not None:
            self._tracer._on_span_ended(self)
        return self

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    # context-manager sugar -----------------------------------------------------

    def __enter__(self) -> Span:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc is not None:
            self.record_exception(exc)
        self.end()


# ---------------------------------------------------------------------------
# Exporter base
# ---------------------------------------------------------------------------


class SpanExporter(ABC):
    """Receives ended spans. Implementations push them to a backend."""

    @abstractmethod
    def export(self, span: Span) -> None:
        """Export a single ended span. Must not raise."""


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """In-process span factory.

    Spans are emitted to every registered exporter when :meth:`Span.end` is
    called. The tracer also keeps a finite in-memory buffer of recent spans
    (useful for tests and the cost dashboard).
    """

    def __init__(
        self,
        exporters: list[SpanExporter] | None = None,
        *,
        buffer_size: int = 1024,
    ) -> None:
        self.exporters: list[SpanExporter] = list(exporters or [])
        self._buffer: list[Span] = []
        self._buffer_size = buffer_size
        self._stack: list[Span] = []  # parent-child stack within a logical scope

    # ----- exporter management -------------------------------------------------

    def add_exporter(self, exporter: SpanExporter) -> None:
        self.exporters.append(exporter)

    def remove_exporter(self, exporter: SpanExporter) -> bool:
        try:
            self.exporters.remove(exporter)
            return True
        except ValueError:
            return False

    # ----- introspection -------------------------------------------------------

    def get_spans(self) -> list[Span]:
        """Return a snapshot of all ended spans in the buffer."""
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._stack.clear()

    @property
    def current_span(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    # ----- span creation -------------------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Span | SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if parent is None and self._stack:
            parent = self._stack[-1]
        if isinstance(parent, Span):
            parent_ctx: SpanContext | None = parent.context
        elif isinstance(parent, SpanContext):
            parent_ctx = parent
        else:
            parent_ctx = None
        ctx = SpanContext(
            trace_id=parent_ctx.trace_id if parent_ctx else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_ctx.span_id if parent_ctx else None,
        )
        span = Span(
            name=name,
            kind=kind,
            context=ctx,
            start_time=time.time(),
            attributes=dict(attributes or {}),
            _tracer=self,
        )
        self._stack.append(span)
        return span

    # ----- internal ------------------------------------------------------------

    def _on_span_ended(self, span: Span) -> None:
        # Pop from stack (best-effort — handles out-of-order ends).
        try:
            self._stack.remove(span)
        except ValueError:
            pass
        if len(self._buffer) >= self._buffer_size:
            self._buffer.pop(0)
        self._buffer.append(span)
        for exporter in self.exporters:
            try:
                exporter.export(span)
            except Exception:
                logger.exception("Span exporter %r failed", type(exporter).__name__)


# ---------------------------------------------------------------------------
# Event-bus wiring
# ---------------------------------------------------------------------------


@dataclass
class _TracerHandle:
    """Returned by :func:`install_tracer`. Call :meth:`uninstall` to detach."""

    tracer: Tracer
    unsubscribers: list[Any]

    def uninstall(self) -> None:
        for unsub in self.unsubscribers:
            try:
                unsub()
            except Exception:  # pragma: no cover - defensive
                pass
        self.unsubscribers.clear()


def install_tracer(event_bus: EventBus, tracer: Tracer | None = None) -> _TracerHandle:
    """Wire ``tracer`` into ``event_bus``.

    Returns a handle whose :meth:`uninstall` removes all hooked handlers.
    """
    t = tracer or Tracer()

    agent_span_holder: dict[str, Span] = {}
    turn_spans: dict[int, Span] = {}
    tool_spans: dict[str, Span] = {}

    def _on_agent_start(event: Any) -> None:
        span = t.start_span(
            "agent.run",
            kind=SpanKind.AGENT,
            attributes={"model": str(getattr(event, "model", ""))},
        )
        agent_span_holder["span"] = span

    def _on_agent_end(event: Any) -> None:
        span = agent_span_holder.pop("span", None)
        if span is None:
            return
        finish = str(getattr(event, "finish_reason", ""))
        span.set_attributes(
            **{
                "agent.total_turns": int(getattr(event, "total_turns", 0)),
                "agent.finish_reason": finish,
            }
        )
        err = getattr(event, "error", None)
        if err:
            span.set_attributes(**{"error": True, "error.message": str(err)})
            span.end(status=SpanStatus.ERROR)
        elif finish == "aborted":
            span.set_attribute("abort", True)
            span.end(status=SpanStatus.ABORTED)
        else:
            span.end(status=SpanStatus.OK)

    def _on_turn_start(event: Any) -> None:
        turn = int(getattr(event, "turn", 0))
        parent = agent_span_holder.get("span")
        span = t.start_span(
            "agent.turn",
            kind=SpanKind.TURN,
            parent=parent,
            attributes={
                "turn.index": turn,
                "turn.message_count": int(getattr(event, "message_count", 0)),
            },
        )
        turn_spans[turn] = span

    def _on_turn_end(event: Any) -> None:
        turn = int(getattr(event, "turn", 0))
        span = turn_spans.pop(turn, None)
        if span is None:
            return
        span.set_attributes(
            **{
                "turn.has_tool_calls": bool(getattr(event, "has_tool_calls", False)),
                "turn.tool_call_count": int(getattr(event, "tool_call_count", 0)),
            }
        )
        # Pull usage/cost off the event if present (set by adapters).
        for attr in ("input_tokens", "output_tokens", "total_tokens", "cost_usd", "cache_hit"):
            value = getattr(event, attr, None)
            if value is not None:
                span.set_attribute(
                    {
                        "input_tokens": "tokens.input",
                        "output_tokens": "tokens.output",
                        "total_tokens": "tokens.total",
                        "cost_usd": "cost_usd",
                        "cache_hit": "cache.hit",
                    }[attr],
                    value,
                )
        span.end()

    def _on_before_tool_call(event: Any) -> None:
        tool_call_id = str(getattr(event, "tool_call_id", ""))
        tool_name = str(getattr(event, "tool_name", ""))
        parent = turn_spans.get(int(getattr(event, "turn", 0))) or agent_span_holder.get("span")
        span = t.start_span(
            "tool.call",
            kind=SpanKind.TOOL,
            parent=parent,
            attributes={
                "tool.name": tool_name,
                "tool.call_id": tool_call_id,
                "turn.index": int(getattr(event, "turn", 0)),
            },
        )
        tool_spans[tool_call_id] = span

    def _on_after_tool_result(event: Any) -> None:
        tool_call_id = str(getattr(event, "tool_call_id", ""))
        span = tool_spans.pop(tool_call_id, None)
        if span is None:
            return
        result = getattr(event, "result", "")
        if isinstance(result, str):
            span.set_attribute("tool.result_length", len(result))
        err = getattr(event, "error", None)
        if err:
            span.set_attributes(**{"error": True, "error.message": str(err)})
            span.end(status=SpanStatus.ERROR)
        else:
            span.end()

    def _on_model_change(event: Any) -> None:
        span = agent_span_holder.get("span")
        if span is None:
            return
        span.add_event(
            "model_change",
            {
                "previous_model": str(getattr(event, "previous_model", "")),
                "new_model": str(getattr(event, "new_model", "")),
            },
        )

    def _on_compaction(event: Any) -> None:
        parent = agent_span_holder.get("span")
        compact_span = t.start_span(
            "compact.run",
            kind=SpanKind.COMPACT,
            parent=parent,
            attributes={
                "messages_before": int(getattr(event, "messages_before", 0) or 0),
                "messages_after": int(getattr(event, "messages_after", 0) or 0),
                "strategy": str(getattr(event, "strategy", "")),
            },
        )
        compact_span.end()

    unsubs = [
        event_bus.on(AGENT_START, _on_agent_start, source="skillengine.tracing"),
        event_bus.on(AGENT_END, _on_agent_end, source="skillengine.tracing"),
        event_bus.on(TURN_START, _on_turn_start, source="skillengine.tracing"),
        event_bus.on(TURN_END, _on_turn_end, source="skillengine.tracing"),
        event_bus.on(BEFORE_TOOL_CALL, _on_before_tool_call, source="skillengine.tracing"),
        event_bus.on(AFTER_TOOL_RESULT, _on_after_tool_result, source="skillengine.tracing"),
        event_bus.on(MODEL_CHANGE, _on_model_change, source="skillengine.tracing"),
        event_bus.on(COMPACTION, _on_compaction, source="skillengine.tracing"),
    ]
    return _TracerHandle(tracer=t, unsubscribers=unsubs)
