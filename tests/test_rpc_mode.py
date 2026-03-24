"""Tests for RPC mode."""

from __future__ import annotations

import asyncio
import json
from io import StringIO
from unittest.mock import MagicMock

import pytest

from skillengine.modes.rpc_mode import RpcMode, RpcResponse


class TestRpcResponse:
    """Tests for the RpcResponse dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        resp = RpcResponse()

        assert resp.id is None
        assert resp.type == "response"
        assert resp.command == ""
        assert resp.success is True
        assert resp.data == {}
        assert resp.error is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        resp = RpcResponse(
            id="req-1",
            type="response",
            command="prompt",
            success=False,
            data={"key": "value"},
            error="something failed",
        )

        assert resp.id == "req-1"
        assert resp.command == "prompt"
        assert resp.success is False
        assert resp.data == {"key": "value"}
        assert resp.error == "something failed"


class TestRpcMode:
    """Tests for RpcMode."""

    def test_constructor_uses_provided_output(self) -> None:
        """Should use the provided output stream."""
        output = StringIO()
        mode = RpcMode(output=output)
        assert mode._output is output

    def test_constructor_uses_provided_input(self) -> None:
        """Should use the provided input stream."""
        input_stream = StringIO()
        mode = RpcMode(input_stream=input_stream)
        assert mode._input is input_stream

    def test_constructor_defaults_to_stdio(self) -> None:
        """Should default to sys.stdout and sys.stdin."""
        import sys

        mode = RpcMode()
        assert mode._output is sys.stdout
        assert mode._input is sys.stdin

    def test_send_writes_json_line(self) -> None:
        """_send should write a JSON line followed by a newline."""
        output = StringIO()
        mode = RpcMode(output=output)

        mode._send({"type": "test", "value": 42})

        line = output.getvalue()
        assert line.endswith("\n")
        parsed = json.loads(line.strip())
        assert parsed["type"] == "test"
        assert parsed["value"] == 42

    def test_send_response_formats_correctly(self) -> None:
        """_send_response should format an RpcResponse as a JSON line."""
        output = StringIO()
        mode = RpcMode(output=output)

        response = RpcResponse(
            id="req-1",
            command="get_state",
            success=True,
            data={"model": "gpt-4"},
        )
        mode._send_response(response)

        parsed = json.loads(output.getvalue().strip())
        assert parsed["type"] == "response"
        assert parsed["command"] == "get_state"
        assert parsed["success"] is True
        assert parsed["id"] == "req-1"
        assert parsed["data"] == {"model": "gpt-4"}

    def test_send_response_omits_none_id(self) -> None:
        """_send_response should not include id when it is None."""
        output = StringIO()
        mode = RpcMode(output=output)

        response = RpcResponse(command="prompt", success=True)
        mode._send_response(response)

        parsed = json.loads(output.getvalue().strip())
        assert "id" not in parsed

    def test_send_response_omits_empty_data(self) -> None:
        """_send_response should not include data when it is empty."""
        output = StringIO()
        mode = RpcMode(output=output)

        response = RpcResponse(command="abort", success=True)
        mode._send_response(response)

        parsed = json.loads(output.getvalue().strip())
        assert "data" not in parsed

    def test_send_response_includes_error_when_present(self) -> None:
        """_send_response should include error when it is set."""
        output = StringIO()
        mode = RpcMode(output=output)

        response = RpcResponse(
            id="req-2",
            command="prompt",
            success=False,
            error="No message provided",
        )
        mode._send_response(response)

        parsed = json.loads(output.getvalue().strip())
        assert parsed["success"] is False
        assert parsed["error"] == "No message provided"

    def test_handle_command_unknown_returns_error(self) -> None:
        """_handle_command should return an error for unknown commands."""
        output = StringIO()
        mode = RpcMode(output=output)

        cmd = {"type": "nonexistent_command", "id": "req-99"}
        asyncio.get_event_loop().run_until_complete(mode._handle_command(cmd))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["success"] is False
        assert parsed["command"] == "nonexistent_command"
        assert "Unknown command" in parsed["error"]

    def test_handle_command_get_state(self) -> None:
        """_handle_command for get_state should return agent state."""
        output = StringIO()
        mode = RpcMode(output=output)

        # Set up a mock agent with config
        mock_config = MagicMock()
        mock_config.model = "claude-3"
        mock_config.thinking_level = "medium"

        mock_agent = MagicMock()
        mock_agent.config = mock_config
        mock_agent.get_history.return_value = [MagicMock(), MagicMock()]

        mode._agent = mock_agent

        cmd = {"type": "get_state", "id": "req-5"}
        asyncio.get_event_loop().run_until_complete(mode._handle_command(cmd))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["success"] is True
        assert parsed["command"] == "get_state"
        assert parsed["id"] == "req-5"
        assert parsed["data"]["model"] == "claude-3"
        assert parsed["data"]["thinking_level"] == "medium"
        assert parsed["data"]["is_streaming"] is False
        assert parsed["data"]["message_count"] == 2

    def test_handle_command_get_state_no_agent(self) -> None:
        """_handle_command for get_state with no agent should return defaults."""
        output = StringIO()
        mode = RpcMode(output=output)
        mode._agent = None

        cmd = {"type": "get_state", "id": "req-6"}
        asyncio.get_event_loop().run_until_complete(mode._handle_command(cmd))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["success"] is True
        assert parsed["data"]["model"] == ""
        assert parsed["data"]["thinking_level"] == "off"
        assert parsed["data"]["is_streaming"] is False
        assert parsed["data"]["message_count"] == 0

    def test_handle_command_unknown_without_id(self) -> None:
        """_handle_command for unknown command without id should still work."""
        output = StringIO()
        mode = RpcMode(output=output)

        cmd = {"type": "bad_command"}
        asyncio.get_event_loop().run_until_complete(mode._handle_command(cmd))

        parsed = json.loads(output.getvalue().strip())
        assert parsed["success"] is False
        assert "id" not in parsed
        assert "Unknown command" in parsed["error"]
