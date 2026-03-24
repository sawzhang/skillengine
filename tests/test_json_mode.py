"""Tests for JSON mode."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillengine.modes.json_mode import JsonMode


def _make_stream_event(
    type: str,
    content: str = "",
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    turn: int = 0,
    error: str | None = None,
    finish_reason: str | None = None,
    args_delta: str | None = None,
    parsed_args: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock StreamEvent with the given fields."""
    event = MagicMock()
    event.type = type
    event.content = content
    event.tool_name = tool_name
    event.tool_call_id = tool_call_id
    event.turn = turn
    event.error = error
    event.finish_reason = finish_reason
    event.args_delta = args_delta
    event.parsed_args = parsed_args
    return event


def _make_mock_agent(events: list[MagicMock]) -> MagicMock:
    """Create a mock agent whose chat_stream_events yields the given events."""
    agent = MagicMock()

    async def fake_stream(prompt: str):
        for event in events:
            yield event

    agent.chat_stream_events = fake_stream
    return agent


class TestJsonMode:
    """Tests for JsonMode."""

    def test_constructor_uses_provided_output(self) -> None:
        """Should use the provided output stream."""
        output = StringIO()
        mode = JsonMode(output=output)
        assert mode._output is output

    def test_constructor_defaults_to_stdout(self) -> None:
        """Should default to sys.stdout when no output is provided."""
        import sys

        mode = JsonMode()
        assert mode._output is sys.stdout

    def test_run_outputs_text_event_as_jsonl(self) -> None:
        """Should output a text_delta event as a JSON line."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [_make_stream_event(type="text_delta", content="Hello")]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "hi"))

        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["type"] == "text_delta"
        assert parsed["content"] == "Hello"

    def test_run_outputs_multiple_events(self) -> None:
        """Should output multiple events, one per line."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [
            _make_stream_event(type="text_start"),
            _make_stream_event(type="text_delta", content="Hi"),
            _make_stream_event(type="text_end"),
        ]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "hello"))

        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["type"] == "text_start"
        assert json.loads(lines[1])["type"] == "text_delta"
        assert json.loads(lines[1])["content"] == "Hi"
        assert json.loads(lines[2])["type"] == "text_end"

    def test_run_includes_tool_name_when_present(self) -> None:
        """Should include tool_name in the output when the event has one."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [
            _make_stream_event(
                type="tool_call_start",
                tool_name="execute",
                tool_call_id="tc1",
            )
        ]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "run ls"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["tool_name"] == "execute"
        assert parsed["tool_call_id"] == "tc1"

    def test_run_includes_turn_when_nonzero(self) -> None:
        """Should include turn in the output when it is nonzero."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [_make_stream_event(type="turn_start", turn=2)]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["turn"] == 2

    def test_run_includes_error_when_present(self) -> None:
        """Should include error in the output when the event has one."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [_make_stream_event(type="error", error="something broke")]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["error"] == "something broke"

    def test_run_includes_finish_reason_when_present(self) -> None:
        """Should include finish_reason in the output when the event has one."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [_make_stream_event(type="done", finish_reason="complete")]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["finish_reason"] == "complete"

    def test_run_includes_args_delta_when_present(self) -> None:
        """Should include args_delta in the output when the event has one."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [
            _make_stream_event(
                type="tool_call_delta",
                tool_name="execute",
                args_delta='{"cmd',
            )
        ]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["args_delta"] == '{"cmd'

    def test_run_includes_parsed_args_when_present(self) -> None:
        """Should include parsed_args in the output when the event has one."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [
            _make_stream_event(
                type="tool_call_delta",
                tool_name="execute",
                parsed_args={"command": "ls"},
            )
        ]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["parsed_args"] == {"command": "ls"}

    def test_run_omits_none_optional_fields(self) -> None:
        """Should not include optional fields when they are None/falsy."""
        output = StringIO()
        mode = JsonMode(output=output)

        events = [_make_stream_event(type="text_delta", content="hi")]
        agent = _make_mock_agent(events)

        asyncio.get_event_loop().run_until_complete(mode.run(agent, "test"))

        parsed = json.loads(output.getvalue().strip())
        assert "tool_name" not in parsed
        assert "tool_call_id" not in parsed
        assert "error" not in parsed
        assert "finish_reason" not in parsed
        assert "args_delta" not in parsed
        assert "parsed_args" not in parsed
