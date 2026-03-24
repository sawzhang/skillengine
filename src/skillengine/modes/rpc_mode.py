"""RPC mode: JSON-line stdin/stdout protocol for programmatic control."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RpcResponse:
    """RPC response sent to stdout."""

    id: str | None = None
    type: str = "response"
    command: str = ""
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RpcMode:
    """RPC mode using JSON-line stdin/stdout protocol.

    Commands (received on stdin):
    - {"type": "prompt", "message": "...", "id": "..."}
    - {"type": "steer", "message": "..."}
    - {"type": "follow_up", "message": "..."}
    - {"type": "abort"}
    - {"type": "new_session"}
    - {"type": "get_state"}
    - {"type": "set_model", "provider": "...", "model_id": "..."}
    - {"type": "set_thinking_level", "level": "..."}
    - {"type": "compact", "custom_instructions": "..."}
    - {"type": "get_messages"}
    - {"type": "get_commands"}
    - {"type": "fork", "entry_id": "..."}
    - {"type": "switch_session", "session_id": "..."}

    Responses (sent to stdout):
    - {"id": "...", "type": "response", "command": "...", "success": true, "data": {...}}
    - {"id": "...", "type": "response", "command": "...", "success": false, "error": "..."}

    Events (sent to stdout during async operations):
    - StreamEvent dicts as JSONL
    """

    def __init__(self, output=None, input_stream=None):
        self._output = output or sys.stdout
        self._input = input_stream or sys.stdin
        self._agent = None
        self._running = False
        self._is_streaming = False

    def _send(self, data: dict[str, Any]) -> None:
        """Send a JSON line to stdout."""
        self._output.write(json.dumps(data) + "\n")
        self._output.flush()

    def _send_response(self, response: RpcResponse) -> None:
        """Send an RPC response."""
        resp_dict: dict[str, Any] = {
            "type": "response",
            "command": response.command,
            "success": response.success,
        }
        if response.id:
            resp_dict["id"] = response.id
        if response.data:
            resp_dict["data"] = response.data
        if response.error:
            resp_dict["error"] = response.error
        self._send(resp_dict)

    def _send_event(self, event: Any) -> None:
        """Send a stream event as JSONL."""
        event_dict: dict[str, Any] = {"type": event.type}
        if event.content:
            event_dict["content"] = event.content
        if event.tool_name:
            event_dict["tool_name"] = event.tool_name
        if event.tool_call_id:
            event_dict["tool_call_id"] = event.tool_call_id
        if event.turn:
            event_dict["turn"] = event.turn
        if event.error:
            event_dict["error"] = event.error
        if event.finish_reason:
            event_dict["finish_reason"] = event.finish_reason
        if event.args_delta:
            event_dict["args_delta"] = event.args_delta
        self._send(event_dict)

    async def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Handle an incoming RPC command."""
        cmd_type = cmd.get("type", "")
        cmd_id = cmd.get("id")

        try:
            if cmd_type == "prompt":
                message = cmd.get("message", "")
                if not message:
                    self._send_response(
                        RpcResponse(
                            id=cmd_id,
                            command="prompt",
                            success=False,
                            error="No message provided",
                        )
                    )
                    return
                self._is_streaming = True
                try:
                    async for event in self._agent.chat_stream_events(message):
                        self._send_event(event)
                finally:
                    self._is_streaming = False
                self._send_response(RpcResponse(id=cmd_id, command="prompt", success=True))

            elif cmd_type == "steer":
                message = cmd.get("message", "")
                if self._agent and message:
                    self._agent.steer(message)
                self._send_response(RpcResponse(id=cmd_id, command="steer", success=True))

            elif cmd_type == "follow_up":
                message = cmd.get("message", "")
                if self._agent and message:
                    self._agent.follow_up(message)
                self._send_response(RpcResponse(id=cmd_id, command="follow_up", success=True))

            elif cmd_type == "abort":
                if self._agent:
                    self._agent.abort()
                self._send_response(RpcResponse(id=cmd_id, command="abort", success=True))

            elif cmd_type == "new_session":
                if self._agent:
                    self._agent.clear_history()
                    self._agent.reset_abort()
                self._send_response(RpcResponse(id=cmd_id, command="new_session", success=True))

            elif cmd_type == "get_state":
                state = {
                    "model": self._agent.config.model if self._agent else "",
                    "thinking_level": (
                        self._agent.config.thinking_level or "off" if self._agent else "off"
                    ),
                    "is_streaming": self._is_streaming,
                    "message_count": (len(self._agent.get_history()) if self._agent else 0),
                }
                self._send_response(
                    RpcResponse(id=cmd_id, command="get_state", success=True, data=state)
                )

            elif cmd_type == "set_model":
                model_id = cmd.get("model_id", "")
                if self._agent and model_id:
                    self._agent.config.model = model_id
                    provider = cmd.get("provider")
                    if provider:
                        self._agent.set_adapter(provider)
                self._send_response(RpcResponse(id=cmd_id, command="set_model", success=True))

            elif cmd_type == "set_thinking_level":
                level = cmd.get("level", "off")
                if self._agent:
                    self._agent.config.thinking_level = level
                self._send_response(
                    RpcResponse(id=cmd_id, command="set_thinking_level", success=True)
                )

            elif cmd_type == "get_messages":
                messages = []
                if self._agent:
                    for msg in self._agent.get_history():
                        messages.append(
                            {
                                "role": msg.role,
                                "content": msg.content,
                                "tool_calls": msg.tool_calls,
                            }
                        )
                self._send_response(
                    RpcResponse(
                        id=cmd_id,
                        command="get_messages",
                        success=True,
                        data={"messages": messages},
                    )
                )

            else:
                self._send_response(
                    RpcResponse(
                        id=cmd_id,
                        command=cmd_type,
                        success=False,
                        error=f"Unknown command: {cmd_type}",
                    )
                )

        except Exception as e:
            self._send_response(
                RpcResponse(id=cmd_id, command=cmd_type, success=False, error=str(e))
            )

    async def run(self, agent: Any) -> None:
        """Run the RPC mode, reading commands from stdin."""
        self._agent = agent
        self._running = True

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, self._input)

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    cmd = json.loads(line_str)
                except json.JSONDecodeError:
                    self._send({"type": "error", "error": "Invalid JSON"})
                    continue
                await self._handle_command(cmd)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._send({"type": "error", "error": str(e)})

    def stop(self) -> None:
        """Stop the RPC mode."""
        self._running = False
