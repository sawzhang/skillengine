"""Tests for agent-level steering, abort, and follow-up (Phase 3)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.agent import (
    AgentAbortedError,
    AgentConfig,
    AgentMessage,
    AgentRunner,
)
from skillengine.engine import SkillsEngine
from skillengine.events import (
    TOOL_EXECUTION_UPDATE,
    EventBus,
    StreamEvent,
    ToolExecutionUpdateEvent,
)
from skillengine.runtime.base import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    events: EventBus | None = None,
    max_turns: int = 5,
) -> AgentRunner:
    """Create an AgentRunner with mocked internals for testing."""
    engine = MagicMock(spec=SkillsEngine)
    engine.extensions = None
    engine.get_snapshot.return_value = MagicMock(skills=[], prompt="", get_skill=lambda n: None)

    config = AgentConfig(
        model="test-model",
        base_url="http://localhost",
        api_key="test-key",
        max_turns=max_turns,
        enable_tools=True,
        auto_execute=True,
    )

    runner = AgentRunner(engine=engine, config=config, events=events or EventBus())
    return runner


def _make_assistant_msg(content: str = "", tool_calls: list[dict] | None = None) -> AgentMessage:
    return AgentMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
    )


def _make_tool_call(tc_id: str = "tc1", name: str = "execute", command: str = "echo hi") -> dict:
    return {
        "id": tc_id,
        "name": name,
        "arguments": json.dumps({"command": command}),
    }


# ---------------------------------------------------------------------------
# AgentAbortedError
# ---------------------------------------------------------------------------


class TestAgentAbortedError:
    def test_is_exception(self) -> None:
        assert issubclass(AgentAbortedError, Exception)

    def test_message(self) -> None:
        err = AgentAbortedError("stopped")
        assert str(err) == "stopped"


# ---------------------------------------------------------------------------
# Abort API
# ---------------------------------------------------------------------------


class TestAbortAPI:
    def test_initially_not_aborted(self) -> None:
        runner = _make_runner()
        assert not runner.is_aborted

    def test_abort_sets_flag(self) -> None:
        runner = _make_runner()
        runner.abort()
        assert runner.is_aborted

    def test_reset_abort_clears_flag(self) -> None:
        runner = _make_runner()
        runner.abort()
        runner.reset_abort()
        assert not runner.is_aborted

    def test_check_abort_raises_when_set(self) -> None:
        runner = _make_runner()
        runner.abort()
        with pytest.raises(AgentAbortedError):
            runner._check_abort()

    def test_check_abort_does_not_raise_when_clear(self) -> None:
        runner = _make_runner()
        runner._check_abort()  # Should not raise


# ---------------------------------------------------------------------------
# Steering API
# ---------------------------------------------------------------------------


class TestSteeringAPI:
    def test_steer_puts_message(self) -> None:
        runner = _make_runner()
        runner.steer("change direction")
        msg = runner._drain_steering()
        assert msg == "change direction"

    def test_drain_steering_returns_none_when_empty(self) -> None:
        runner = _make_runner()
        assert runner._drain_steering() is None

    def test_multiple_steers_fifo(self) -> None:
        runner = _make_runner()
        runner.steer("first")
        runner.steer("second")
        assert runner._drain_steering() == "first"
        assert runner._drain_steering() == "second"
        assert runner._drain_steering() is None


# ---------------------------------------------------------------------------
# Follow-up API
# ---------------------------------------------------------------------------


class TestFollowUpAPI:
    def test_follow_up_puts_message(self) -> None:
        runner = _make_runner()
        runner.follow_up("next task")
        msg = runner._drain_followup()
        assert msg == "next task"

    def test_drain_followup_returns_none_when_empty(self) -> None:
        runner = _make_runner()
        assert runner._drain_followup() is None


# ---------------------------------------------------------------------------
# chat() with abort
# ---------------------------------------------------------------------------


class TestChatAbort:
    @pytest.mark.asyncio
    async def test_abort_before_chat_returns_aborted(self) -> None:
        runner = _make_runner()
        runner.abort()

        # _call_llm should not be called
        runner._call_llm = AsyncMock()

        result = await runner.chat("hello")
        assert result.content == "[Aborted]"
        runner._call_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_during_tool_execution(self) -> None:
        """Abort between tool calls skips remaining tools."""
        runner = _make_runner()

        call_count = 0

        async def mock_call_llm(messages, stream=False):
            return _make_assistant_msg(
                tool_calls=[
                    _make_tool_call("tc1", command="echo first"),
                    _make_tool_call("tc2", command="echo second"),
                ]
            )

        async def mock_execute_tool(tool_call, on_output=None):
            nonlocal call_count
            call_count += 1
            # Abort after first tool
            runner.abort()
            return "ok"

        runner._call_llm = mock_call_llm
        runner._execute_tool = mock_execute_tool

        result = await runner.chat("do stuff")
        assert result.content == "[Aborted]"
        # Only one tool should have been executed (abort before second)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_abort_finish_reason_in_agent_end(self) -> None:
        """Agent end event should have finish_reason='aborted'."""
        events = EventBus()
        runner = _make_runner(events=events)
        runner.abort()

        end_events: list[Any] = []
        events.on("agent_end", lambda e: end_events.append(e))

        await runner.chat("hello")

        assert len(end_events) == 1
        assert end_events[0].finish_reason == "aborted"


# ---------------------------------------------------------------------------
# chat() with steering
# ---------------------------------------------------------------------------


class TestChatSteering:
    @pytest.mark.asyncio
    async def test_steering_breaks_tool_chain(self) -> None:
        """Steering message between tools should stop remaining tools
        and start a new turn."""
        runner = _make_runner(max_turns=3)

        turn_count = 0

        async def mock_call_llm(messages, stream=False):
            nonlocal turn_count
            turn_count += 1

            if turn_count == 1:
                # First turn: return two tool calls
                return _make_assistant_msg(
                    tool_calls=[
                        _make_tool_call("tc1", command="echo first"),
                        _make_tool_call("tc2", command="echo second"),
                    ]
                )
            elif turn_count == 2:
                # Second turn: after steering, return text
                return _make_assistant_msg(content="steered response")
            return _make_assistant_msg(content="done")

        tool_calls_executed: list[str] = []

        async def mock_execute_tool(tool_call, on_output=None):
            tc_id = tool_call["id"]
            tool_calls_executed.append(tc_id)
            # After first tool, inject steering
            if tc_id == "tc1":
                runner.steer("change plan")
            return "ok"

        runner._call_llm = mock_call_llm
        runner._execute_tool = mock_execute_tool

        result = await runner.chat("do stuff")

        # Only tc1 should have executed — tc2 skipped due to steering
        assert tool_calls_executed == ["tc1"]
        assert result.content == "steered response"
        assert turn_count == 2

    @pytest.mark.asyncio
    async def test_steering_at_turn_start(self) -> None:
        """Steering injected before a turn starts should be added to conversation."""
        runner = _make_runner(max_turns=2)

        turn_count = 0

        async def mock_call_llm(messages, stream=False):
            nonlocal turn_count
            turn_count += 1
            return _make_assistant_msg(content=f"response {turn_count}")

        runner._call_llm = mock_call_llm

        # Pre-inject steering
        runner.steer("extra context")

        result = await runner.chat("hello")
        # Steering is drained at start of turn 0, but since there are no tool calls,
        # the loop exits after the first LLM response
        assert result.content == "response 1"
        # The steering message should be in the conversation
        user_msgs = [m for m in runner._conversation if m.role == "user"]
        contents = [m.content for m in user_msgs]
        assert "hello" in contents
        assert "extra context" in contents


# ---------------------------------------------------------------------------
# chat_stream_events() with abort
# ---------------------------------------------------------------------------


class TestStreamEventsAbort:
    @pytest.mark.asyncio
    async def test_stream_abort_yields_done_aborted(self) -> None:
        runner = _make_runner()
        runner.abort()
        runner._call_llm = AsyncMock()

        events: list[StreamEvent] = []
        async for event in runner.chat_stream_events("hello"):
            events.append(event)

        # Should have done event with finish_reason="aborted"
        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1
        assert done_events[0].finish_reason == "aborted"


# ---------------------------------------------------------------------------
# chat_stream_events() with tool_output
# ---------------------------------------------------------------------------


class TestStreamEventsToolOutput:
    @pytest.mark.asyncio
    async def test_tool_output_events_yielded(self) -> None:
        """Tool streaming output should produce tool_output StreamEvents."""
        runner = _make_runner(max_turns=2)

        turn_count = 0

        async def mock_call_llm_stream(messages):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                # First turn: one tool call
                yield StreamEvent(type="tool_call_start", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(type="tool_call_delta", tool_call_id="tc1", tool_name="execute", args_delta='{"command":"echo hi"}')
                yield StreamEvent(type="tool_call_end", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(type="done", finish_reason="tool_calls")
            else:
                # Second turn: text response
                yield StreamEvent(type="text_start")
                yield StreamEvent(type="text_delta", content="done")
                yield StreamEvent(type="text_end")
                yield StreamEvent(type="done", finish_reason="stop")

        async def mock_execute_tool(tool_call, on_output=None):
            # Simulate streaming output
            if on_output:
                on_output("line1\n")
                on_output("line2\n")
            return "line1\nline2\n"

        runner._call_llm_stream = mock_call_llm_stream
        runner._execute_tool = mock_execute_tool

        events: list[StreamEvent] = []
        async for event in runner.chat_stream_events("run it"):
            events.append(event)

        # Check for tool_output events
        tool_outputs = [e for e in events if e.type == "tool_output"]
        assert len(tool_outputs) == 2
        assert tool_outputs[0].content == "line1\n"
        assert tool_outputs[1].content == "line2\n"
        assert tool_outputs[0].tool_call_id == "tc1"
        assert tool_outputs[0].tool_name == "execute"


# ---------------------------------------------------------------------------
# TOOL_EXECUTION_UPDATE EventBus events
# ---------------------------------------------------------------------------


class TestToolExecutionUpdateEvent:
    def test_event_constant(self) -> None:
        assert TOOL_EXECUTION_UPDATE == "tool_execution_update"

    def test_event_dataclass(self) -> None:
        event = ToolExecutionUpdateEvent(
            tool_call_id="tc1",
            tool_name="execute",
            output="hello\n",
            turn=0,
        )
        assert event.tool_call_id == "tc1"
        assert event.tool_name == "execute"
        assert event.output == "hello\n"
        assert event.turn == 0

    @pytest.mark.asyncio
    async def test_tool_update_event_fired_in_chat(self) -> None:
        """chat() should emit TOOL_EXECUTION_UPDATE via EventBus."""
        events = EventBus()
        runner = _make_runner(events=events, max_turns=2)

        received: list[ToolExecutionUpdateEvent] = []
        events.on(TOOL_EXECUTION_UPDATE, lambda e: received.append(e))

        turn_count = 0

        async def mock_call_llm(messages, stream=False):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                return _make_assistant_msg(
                    tool_calls=[_make_tool_call("tc1", command="echo test")]
                )
            return _make_assistant_msg(content="done")

        original_execute_tool = runner._execute_tool

        async def mock_execute_tool(tool_call, on_output=None):
            # Simulate on_output callback being fired
            if on_output:
                on_output("test output\n")
            return "test output\n"

        runner._call_llm = mock_call_llm
        runner._execute_tool = mock_execute_tool

        result = await runner.chat("test")

        # Allow fire-and-forget tasks to complete
        await asyncio.sleep(0.1)

        assert len(received) >= 1
        assert received[0].output == "test output\n"
        assert received[0].tool_call_id == "tc1"
        assert received[0].tool_name == "execute"


# ---------------------------------------------------------------------------
# Agent-level _execute_tool passes abort_signal
# ---------------------------------------------------------------------------


class TestExecuteToolIntegration:
    @pytest.mark.asyncio
    async def test_execute_tool_passes_abort_signal(self) -> None:
        """_execute_tool should pass _abort_event to engine.execute."""
        runner = _make_runner()

        # Mock engine.execute to capture call args
        captured_kwargs: dict[str, Any] = {}

        async def mock_engine_execute(command, **kwargs):
            captured_kwargs.update(kwargs)
            return ExecutionResult.success_result(output="ok")

        runner.engine.execute = mock_engine_execute

        tool_call = _make_tool_call("tc1", command="echo test")
        result = await runner._execute_tool(tool_call)

        assert result == "ok"
        assert "abort_signal" in captured_kwargs
        assert captured_kwargs["abort_signal"] is runner._abort_event

    @pytest.mark.asyncio
    async def test_execute_tool_passes_on_output(self) -> None:
        """_execute_tool should pass on_output callback to engine.execute."""
        runner = _make_runner()

        captured_kwargs: dict[str, Any] = {}

        async def mock_engine_execute(command, **kwargs):
            captured_kwargs.update(kwargs)
            # Simulate calling on_output
            if kwargs.get("on_output"):
                kwargs["on_output"]("line\n")
            return ExecutionResult.success_result(output="line\n")

        runner.engine.execute = mock_engine_execute

        lines: list[str] = []
        tool_call = _make_tool_call("tc1", command="echo test")
        result = await runner._execute_tool(tool_call, on_output=lambda l: lines.append(l))

        assert "on_output" in captured_kwargs
        assert lines == ["line\n"]

    @pytest.mark.asyncio
    async def test_execute_script_tool(self) -> None:
        """_execute_tool should handle execute_script tool."""
        runner = _make_runner()

        async def mock_engine_execute_script(script, **kwargs):
            return ExecutionResult.success_result(output="script output")

        runner.engine.execute_script = mock_engine_execute_script

        tool_call = {
            "id": "tc1",
            "name": "execute_script",
            "arguments": json.dumps({"script": "echo hello\nexit 0"}),
        }
        result = await runner._execute_tool(tool_call)
        assert result == "script output"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self) -> None:
        runner = _make_runner()
        tool_call = {
            "id": "tc1",
            "name": "unknown_tool",
            "arguments": "{}",
        }
        result = await runner._execute_tool(tool_call)
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# Imports / exports
# ---------------------------------------------------------------------------


class TestPhase3Exports:
    def test_agent_aborted_error_exported(self) -> None:
        from skillengine import AgentAbortedError
        assert AgentAbortedError is not None

    def test_tool_execution_update_exported(self) -> None:
        from skillengine import TOOL_EXECUTION_UPDATE
        assert TOOL_EXECUTION_UPDATE == "tool_execution_update"

    def test_tool_execution_update_event_exported(self) -> None:
        from skillengine import ToolExecutionUpdateEvent
        assert ToolExecutionUpdateEvent is not None
