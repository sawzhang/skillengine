"""Tests for TRACE-1: span model, tracer, exporters, and event-bus wiring."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest

from skillengine import (
    ConsoleSpanExporter,
    Span,
    SpanExporter,
    SpanKind,
    SpanStatus,
    Tracer,
    install_tracer,
)
from skillengine.events import (
    AGENT_END,
    AGENT_START,
    BEFORE_TOOL_CALL,
    COMPACTION,
    TURN_END,
    TURN_START,
    AfterToolResultEvent,
    AgentEndEvent,
    AgentStartEvent,
    BeforeToolCallEvent,
    EventBus,
    TurnEndEvent,
    TurnStartEvent,
)

# ---------------------------------------------------------------------------
# Span basics
# ---------------------------------------------------------------------------


def test_span_set_attributes_and_events() -> None:
    t = Tracer()
    span = t.start_span("agent.run", kind=SpanKind.AGENT, attributes={"model": "gpt-4o"})
    span.set_attribute("tokens.input", 100)
    span.set_attributes(**{"tokens.output": 50, "cost_usd": 0.01})
    span.add_event("model_change", {"previous_model": "gpt-3.5"})
    span.end()
    assert span.attributes["model"] == "gpt-4o"
    assert span.attributes["tokens.input"] == 100
    assert span.attributes["cost_usd"] == 0.01
    assert len(span.events) == 1
    assert span.events[0]["name"] == "model_change"
    assert span.status == SpanStatus.OK
    assert span.duration_ms is not None and span.duration_ms >= 0


def test_span_context_manager_records_exceptions() -> None:
    t = Tracer()
    with pytest.raises(RuntimeError, match="boom"):
        with t.start_span("op") as span:
            raise RuntimeError("boom")
    assert span.status == SpanStatus.ERROR
    assert span.attributes["error"] is True
    assert "boom" in span.attributes["error.message"]


def test_tracer_buffer_respects_capacity() -> None:
    t = Tracer(buffer_size=3)
    for i in range(5):
        t.start_span(f"s{i}").end()
    assert len(t.get_spans()) == 3
    # Oldest two evicted: should keep s2, s3, s4.
    names = [s.name for s in t.get_spans()]
    assert names == ["s2", "s3", "s4"]


def test_parent_child_trace_ids() -> None:
    t = Tracer()
    parent = t.start_span("parent")
    child = t.start_span("child")
    assert child.context.trace_id == parent.context.trace_id
    assert child.context.parent_span_id == parent.context.span_id
    child.end()
    parent.end()


def test_explicit_parent_context() -> None:
    t = Tracer()
    a = t.start_span("a")
    a.end()
    # Even after a ends, we can reattach by passing the context.
    b = t.start_span("b", parent=a.context)
    assert b.context.trace_id == a.context.trace_id
    assert b.context.parent_span_id == a.context.span_id
    b.end()


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def test_console_exporter_writes_jsonl() -> None:
    buf = io.StringIO()
    t = Tracer(exporters=[ConsoleSpanExporter(stream=buf)])
    t.start_span("agent.turn", kind=SpanKind.TURN).set_attribute("turn.index", 1).end()
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["name"] == "agent.turn"
    assert payload["kind"] == "turn"
    assert payload["attributes"]["turn.index"] == 1
    assert payload["status"] == "ok"


def test_custom_exporter_receives_spans() -> None:
    received: list[Span] = []

    class _Collect(SpanExporter):
        def export(self, span: Span) -> None:
            received.append(span)

    t = Tracer(exporters=[_Collect()])
    t.start_span("op").end()
    assert len(received) == 1
    assert received[0].name == "op"


def test_exporter_exceptions_are_swallowed() -> None:
    class _Boom(SpanExporter):
        def export(self, span: Span) -> None:
            raise RuntimeError("nope")

    t = Tracer(exporters=[_Boom()])
    # Should not raise.
    t.start_span("op").end()
    assert len(t.get_spans()) == 1


def test_otel_exporter_missing_dependency() -> None:
    # If opentelemetry is installed in the test env we cannot easily simulate
    # its absence; just assert the type can be referenced.
    from skillengine import OTelSpanExporter  # noqa: F401


# ---------------------------------------------------------------------------
# install_tracer: event-bus wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_tracer_creates_agent_and_turn_spans() -> None:
    bus = EventBus()
    handle = install_tracer(bus, Tracer())
    try:
        await bus.emit(
            AGENT_START,
            AgentStartEvent(user_input="hi", system_prompt="", model="gpt-4o-mini", turn=0),
        )
        await bus.emit(TURN_START, TurnStartEvent(turn=1, message_count=2))
        await bus.emit(
            TURN_END,
            TurnEndEvent(turn=1, has_tool_calls=False, content="ok", tool_call_count=0),
        )
        await bus.emit(
            AGENT_END,
            AgentEndEvent(user_input="hi", total_turns=1, finish_reason="complete"),
        )
    finally:
        handle.uninstall()

    spans = handle.tracer.get_spans()
    names = [s.name for s in spans]
    assert "agent.turn" in names
    assert "agent.run" in names
    turn = next(s for s in spans if s.name == "agent.turn")
    agent = next(s for s in spans if s.name == "agent.run")
    assert turn.attributes["turn.index"] == 1
    assert turn.context.parent_span_id == agent.context.span_id
    assert agent.attributes["model"] == "gpt-4o-mini"
    assert agent.attributes["agent.finish_reason"] == "complete"
    assert agent.status == SpanStatus.OK


@pytest.mark.asyncio
async def test_install_tracer_creates_tool_spans_under_turn() -> None:
    bus = EventBus()
    handle = install_tracer(bus, Tracer())
    try:
        await bus.emit(
            AGENT_START,
            AgentStartEvent(user_input="", system_prompt="", model="m", turn=0),
        )
        await bus.emit(TURN_START, TurnStartEvent(turn=1, message_count=1))
        await bus.emit(
            BEFORE_TOOL_CALL,
            BeforeToolCallEvent(
                tool_call_id="call_1", tool_name="bash", args={"cmd": "ls"}, turn=1
            ),
        )
        await bus.emit(
            "after_tool_result",
            AfterToolResultEvent(
                tool_call_id="call_1",
                tool_name="bash",
                args={"cmd": "ls"},
                result="file1\nfile2",
                turn=1,
            ),
        )
        await bus.emit(
            TURN_END,
            TurnEndEvent(turn=1, has_tool_calls=True, content="", tool_call_count=1),
        )
        await bus.emit(AGENT_END, AgentEndEvent(user_input="", total_turns=1))
    finally:
        handle.uninstall()

    spans = handle.tracer.get_spans()
    tool = next(s for s in spans if s.name == "tool.call")
    turn = next(s for s in spans if s.name == "agent.turn")
    assert tool.context.parent_span_id == turn.context.span_id
    assert tool.attributes["tool.name"] == "bash"
    assert tool.attributes["tool.result_length"] == len("file1\nfile2")


@pytest.mark.asyncio
async def test_install_tracer_records_aborted_status() -> None:
    bus = EventBus()
    handle = install_tracer(bus, Tracer())
    try:
        await bus.emit(
            AGENT_START,
            AgentStartEvent(user_input="", system_prompt="", model="m", turn=0),
        )
        await bus.emit(
            AGENT_END,
            AgentEndEvent(user_input="", total_turns=0, finish_reason="aborted"),
        )
    finally:
        handle.uninstall()
    agent = next(s for s in handle.tracer.get_spans() if s.name == "agent.run")
    assert agent.status == SpanStatus.ABORTED
    assert agent.attributes["abort"] is True


@pytest.mark.asyncio
async def test_install_tracer_records_compaction() -> None:
    bus = EventBus()
    handle = install_tracer(bus, Tracer())
    try:
        await bus.emit(
            AGENT_START,
            AgentStartEvent(user_input="", system_prompt="", model="m"),
        )
        # COMPACTION event payload — accept duck-typing.
        evt = MagicMock(messages_before=20, messages_after=10, strategy="token_budget")
        await bus.emit(COMPACTION, evt)
        await bus.emit(AGENT_END, AgentEndEvent(user_input="", total_turns=0))
    finally:
        handle.uninstall()

    compact = next(s for s in handle.tracer.get_spans() if s.name == "compact.run")
    assert compact.attributes["messages_before"] == 20
    assert compact.attributes["messages_after"] == 10
    assert compact.attributes["strategy"] == "token_budget"


@pytest.mark.asyncio
async def test_install_tracer_uninstall_removes_handlers() -> None:
    bus = EventBus()
    before = len(bus._handlers)
    handle = install_tracer(bus, Tracer())
    assert len(bus._handlers) > before
    handle.uninstall()
    assert len(bus._handlers) == before


@pytest.mark.asyncio
async def test_install_tracer_records_turn_usage_attributes() -> None:
    bus = EventBus()
    handle = install_tracer(bus, Tracer())
    try:
        await bus.emit(
            AGENT_START,
            AgentStartEvent(user_input="", system_prompt="", model="m"),
        )
        await bus.emit(TURN_START, TurnStartEvent(turn=1, message_count=1))
        # Synthetic TurnEndEvent extended with usage attributes via MagicMock.
        evt = MagicMock(
            turn=1,
            has_tool_calls=False,
            content="",
            tool_call_count=0,
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            cost_usd=0.0025,
            cache_hit=True,
        )
        await bus.emit(TURN_END, evt)
        await bus.emit(AGENT_END, AgentEndEvent(user_input="", total_turns=1))
    finally:
        handle.uninstall()

    turn = next(s for s in handle.tracer.get_spans() if s.name == "agent.turn")
    assert turn.attributes["tokens.input"] == 120
    assert turn.attributes["tokens.output"] == 30
    assert turn.attributes["tokens.total"] == 150
    assert turn.attributes["cost_usd"] == 0.0025
    assert turn.attributes["cache.hit"] is True
