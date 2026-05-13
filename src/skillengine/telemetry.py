"""OpenTelemetry tracing integration for skillengine.

Wires :class:`~skillengine.events.EventBus` events into OpenTelemetry spans so
that operators can see agent turns, tool calls, and model changes inside any
OTLP-compatible backend (Jaeger, Tempo, Honeycomb, Grafana Cloud, etc.).

This module degrades gracefully when ``opentelemetry-api`` is **not**
installed: importing it succeeds, but :func:`install` raises a clear
``RuntimeError`` instructing the user to ``pip install
skillengine[telemetry]``. Calling code can therefore always import this
module unconditionally.

Example::

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    from skillengine.telemetry import install as install_otel
    handle = install_otel(agent.events)
    try:
        await agent.chat("hello")
    finally:
        handle.uninstall()

Only the eight lifecycle events that matter for tracing are wired:
``agent_start``/``agent_end`` (root span), ``turn_start``/``turn_end``
(per-turn spans), ``before_tool_call``/``after_tool_result`` (per-tool
spans), and ``model_change`` (span event).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skillengine.events import (
    AFTER_TOOL_RESULT,
    AGENT_END,
    AGENT_START,
    BEFORE_TOOL_CALL,
    MODEL_CHANGE,
    TURN_END,
    TURN_START,
)

if TYPE_CHECKING:
    from skillengine.events import EventBus


_SOURCE = "skillengine.telemetry"


def _require_otel() -> Any:
    """Return the ``opentelemetry.trace`` module or raise a clear error."""
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised by users without otel
        raise RuntimeError(
            "OpenTelemetry is not installed. Install the extra: "
            "`pip install skillengine[telemetry]` (provides "
            "opentelemetry-api, opentelemetry-sdk)."
        ) from e
    return trace


@dataclass
class TelemetryHandle:
    """Returned by :func:`install`. Call :meth:`uninstall` to detach."""

    event_bus: EventBus
    source: str = _SOURCE

    def uninstall(self) -> int:
        """Remove all handlers registered by this install. Returns count."""
        return self.event_bus.off_by_source(self.source)


def install(
    event_bus: EventBus,
    tracer_name: str = "skillengine",
    *,
    capture_tool_args: bool = False,
    capture_tool_results: bool = False,
) -> TelemetryHandle:
    """Wire OpenTelemetry tracing into ``event_bus``.

    Args:
        event_bus: The agent's :class:`EventBus`.
        tracer_name: Name passed to ``trace.get_tracer``.
        capture_tool_args: If True, attach tool arguments as span attributes.
            Off by default since arguments may contain sensitive content
            (file paths, prompts, secrets in commands).
        capture_tool_results: If True, attach tool result strings (truncated to
            512 chars) as span events. Off by default for the same reasons.

    Returns:
        A :class:`TelemetryHandle` that can :meth:`uninstall` the wiring.
    """
    trace = _require_otel()
    tracer = trace.get_tracer(tracer_name)

    # Maps to track open spans across paired start/end events. We use plain
    # dicts (not weakref) because the dict only holds keys for the in-flight
    # turn / tool call. The matching *_end / after_* handler always pops them.
    agent_span_ctx: dict[str, Any] = {}
    turn_spans: dict[int, Any] = {}
    tool_spans: dict[str, Any] = {}

    def _on_agent_start(event: Any) -> None:
        span = tracer.start_span("skillengine.agent")
        span.set_attribute("agent.model", str(getattr(event, "model", "")))
        agent_span_ctx["span"] = span
        agent_span_ctx["ctx_mgr"] = trace.use_span(span, end_on_exit=False)
        agent_span_ctx["ctx_mgr"].__enter__()

    def _on_agent_end(event: Any) -> None:
        span = agent_span_ctx.pop("span", None)
        ctx_mgr = agent_span_ctx.pop("ctx_mgr", None)
        if span is None:
            return
        try:
            span.set_attribute("agent.total_turns", int(getattr(event, "total_turns", 0)))
            finish_reason = str(getattr(event, "finish_reason", ""))
            if finish_reason:
                span.set_attribute("agent.finish_reason", finish_reason)
            err = getattr(event, "error", None)
            if err:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(err))
        finally:
            if ctx_mgr is not None:
                ctx_mgr.__exit__(None, None, None)
            span.end()

    def _on_turn_start(event: Any) -> None:
        turn = int(getattr(event, "turn", 0))
        span = tracer.start_span(f"agent.turn.{turn}")
        span.set_attribute("turn.index", turn)
        span.set_attribute("turn.message_count", int(getattr(event, "message_count", 0)))
        turn_spans[turn] = span

    def _on_turn_end(event: Any) -> None:
        turn = int(getattr(event, "turn", 0))
        span = turn_spans.pop(turn, None)
        if span is None:
            return
        span.set_attribute(
            "turn.has_tool_calls",
            bool(getattr(event, "has_tool_calls", False)),
        )
        span.set_attribute(
            "turn.tool_call_count",
            int(getattr(event, "tool_call_count", 0)),
        )
        span.end()

    def _on_before_tool_call(event: Any) -> None:
        tool_call_id = str(getattr(event, "tool_call_id", ""))
        tool_name = str(getattr(event, "tool_name", ""))
        span = tracer.start_span(f"tool.{tool_name}")
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.call_id", tool_call_id)
        span.set_attribute("turn.index", int(getattr(event, "turn", 0)))
        if capture_tool_args:
            args = getattr(event, "args", {}) or {}
            for k, v in args.items():
                span.set_attribute(f"tool.arg.{k}", _safe_value(v))
        tool_spans[tool_call_id] = span

    def _on_after_tool_result(event: Any) -> None:
        tool_call_id = str(getattr(event, "tool_call_id", ""))
        span = tool_spans.pop(tool_call_id, None)
        if span is None:
            return
        if capture_tool_results:
            result = str(getattr(event, "result", ""))[:512]
            span.add_event("tool.result", {"value": result})
        span.end()

    def _on_model_change(event: Any) -> None:
        span = agent_span_ctx.get("span")
        if span is None:
            return
        span.add_event(
            "model_change",
            {
                "previous_model": str(getattr(event, "previous_model", "")),
                "new_model": str(getattr(event, "new_model", "")),
            },
        )

    event_bus.on(AGENT_START, _on_agent_start, source=_SOURCE)
    event_bus.on(AGENT_END, _on_agent_end, source=_SOURCE)
    event_bus.on(TURN_START, _on_turn_start, source=_SOURCE)
    event_bus.on(TURN_END, _on_turn_end, source=_SOURCE)
    event_bus.on(BEFORE_TOOL_CALL, _on_before_tool_call, source=_SOURCE)
    event_bus.on(AFTER_TOOL_RESULT, _on_after_tool_result, source=_SOURCE)
    event_bus.on(MODEL_CHANGE, _on_model_change, source=_SOURCE)

    return TelemetryHandle(event_bus=event_bus)


def _safe_value(v: Any) -> str | int | float | bool:
    """Coerce a tool-argument value to an OTel-compatible attribute type."""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)[:512]


__all__ = ["TelemetryHandle", "install"]
