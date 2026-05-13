"""Tests for the optional OpenTelemetry integration."""

from __future__ import annotations

import importlib.util

import pytest

from skillengine.events import (
    AFTER_TOOL_RESULT,
    AGENT_END,
    AGENT_START,
    BEFORE_TOOL_CALL,
    MODEL_CHANGE,
    TURN_END,
    TURN_START,
    AfterToolResultEvent,
    AgentEndEvent,
    AgentStartEvent,
    BeforeToolCallEvent,
    EventBus,
    ModelChangeEvent,
    TurnEndEvent,
    TurnStartEvent,
)

_HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_OTEL,
    reason="opentelemetry-api not installed; install with `skillengine[telemetry]`",
)


@pytest.fixture(scope="module")
def _otel_provider() -> object:
    """Set the global tracer provider once per module (OTel disallows resets)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def in_memory_tracer(_otel_provider: object) -> object:
    """Reset the in-memory exporter at the start of each test."""
    _otel_provider.clear()  # type: ignore[attr-defined]
    return _otel_provider


async def test_install_returns_handle_and_uninstalls(in_memory_tracer: object) -> None:
    from skillengine.telemetry import install

    bus = EventBus()
    before = len(bus._handlers)
    handle = install(bus)
    assert len(bus._handlers) > before
    removed = handle.uninstall()
    assert removed > 0
    assert len(bus._handlers) == before


async def test_emits_spans_for_agent_lifecycle(in_memory_tracer: object) -> None:
    from skillengine.telemetry import install

    bus = EventBus()
    install(bus)

    await bus.emit(
        AGENT_START,
        AgentStartEvent(user_input="hi", system_prompt="sys", model="gpt-4"),
    )
    await bus.emit(TURN_START, TurnStartEvent(turn=0, message_count=2))
    await bus.emit(
        BEFORE_TOOL_CALL,
        BeforeToolCallEvent(
            tool_call_id="call_1",
            tool_name="bash",
            args={"command": "ls"},
            turn=0,
        ),
    )
    await bus.emit(
        AFTER_TOOL_RESULT,
        AfterToolResultEvent(
            tool_call_id="call_1",
            tool_name="bash",
            args={"command": "ls"},
            result="ok",
            turn=0,
        ),
    )
    await bus.emit(
        TURN_END,
        TurnEndEvent(turn=0, has_tool_calls=True, tool_call_count=1),
    )
    await bus.emit(
        AGENT_END,
        AgentEndEvent(user_input="hi", total_turns=1, finish_reason="complete"),
    )

    spans = in_memory_tracer.get_finished_spans()  # type: ignore[attr-defined]
    names = [s.name for s in spans]
    assert "skillengine.agent" in names
    assert "agent.turn.0" in names
    assert "tool.bash" in names


async def test_model_change_recorded_as_span_event(in_memory_tracer: object) -> None:
    from skillengine.telemetry import install

    bus = EventBus()
    install(bus)

    await bus.emit(
        AGENT_START,
        AgentStartEvent(user_input="hi", system_prompt="", model="gpt-4"),
    )
    await bus.emit(
        MODEL_CHANGE,
        ModelChangeEvent(previous_model="gpt-4", new_model="claude-3"),
    )
    await bus.emit(
        AGENT_END,
        AgentEndEvent(user_input="hi", total_turns=0, finish_reason="complete"),
    )

    spans = in_memory_tracer.get_finished_spans()  # type: ignore[attr-defined]
    agent_spans = [s for s in spans if s.name == "skillengine.agent"]
    assert agent_spans, "agent root span missing"
    event_names = [e.name for e in agent_spans[0].events]
    assert "model_change" in event_names


async def test_capture_tool_args_off_by_default(in_memory_tracer: object) -> None:
    from skillengine.telemetry import install

    bus = EventBus()
    install(bus)  # capture_tool_args defaults to False

    await bus.emit(
        AGENT_START,
        AgentStartEvent(user_input="x", system_prompt="", model="m"),
    )
    await bus.emit(
        BEFORE_TOOL_CALL,
        BeforeToolCallEvent(
            tool_call_id="c1",
            tool_name="bash",
            args={"command": "secret"},
            turn=0,
        ),
    )
    await bus.emit(
        AFTER_TOOL_RESULT,
        AfterToolResultEvent(
            tool_call_id="c1",
            tool_name="bash",
            args={"command": "secret"},
            result="hidden",
            turn=0,
        ),
    )
    await bus.emit(
        AGENT_END,
        AgentEndEvent(user_input="x", total_turns=1, finish_reason="complete"),
    )

    spans = in_memory_tracer.get_finished_spans()  # type: ignore[attr-defined]
    tool_spans = [s for s in spans if s.name == "tool.bash"]
    assert tool_spans
    # By default, the command argument is not exported.
    assert "tool.arg.command" not in tool_spans[0].attributes


async def test_capture_tool_args_when_enabled(in_memory_tracer: object) -> None:
    from skillengine.telemetry import install

    bus = EventBus()
    install(bus, capture_tool_args=True)

    await bus.emit(
        AGENT_START,
        AgentStartEvent(user_input="x", system_prompt="", model="m"),
    )
    await bus.emit(
        BEFORE_TOOL_CALL,
        BeforeToolCallEvent(
            tool_call_id="c1",
            tool_name="bash",
            args={"command": "ls /tmp"},
            turn=0,
        ),
    )
    await bus.emit(
        AFTER_TOOL_RESULT,
        AfterToolResultEvent(
            tool_call_id="c1",
            tool_name="bash",
            args={"command": "ls /tmp"},
            result="ok",
            turn=0,
        ),
    )
    await bus.emit(
        AGENT_END,
        AgentEndEvent(user_input="x", total_turns=1, finish_reason="complete"),
    )

    spans = in_memory_tracer.get_finished_spans()  # type: ignore[attr-defined]
    tool_spans = [s for s in spans if s.name == "tool.bash"]
    assert tool_spans
    assert tool_spans[0].attributes.get("tool.arg.command") == "ls /tmp"
