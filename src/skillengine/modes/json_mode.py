"""JSON mode: outputs stream events as JSONL to stdout."""

from __future__ import annotations

import json
import sys
from typing import Any


class JsonMode:
    """Single-shot mode that outputs all StreamEvents as JSONL to stdout.

    Usage:
        mode = JsonMode()
        await mode.run(agent, "prompt text")
    """

    def __init__(self, output=None):
        self._output = output or sys.stdout

    async def run(self, agent: Any, prompt: str) -> None:
        """Run agent with prompt, outputting events as JSONL."""
        async for event in agent.chat_stream_events(prompt):
            event_dict = {
                "type": event.type,
                "content": event.content,
            }
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
            if event.parsed_args:
                event_dict["parsed_args"] = event.parsed_args

            self._output.write(json.dumps(event_dict) + "\n")
            self._output.flush()
