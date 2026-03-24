"""
OpenAI adapter for the skills engine.

Requires the 'openai' extra: pip install skillengine[openai]
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

try:
    from openai import AsyncOpenAI  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "OpenAI adapter requires the 'openai' package. Install with: pip install skillengine[openai]"
    )

from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
from skillengine.engine import SkillsEngine
from skillengine.events import StreamEvent
from skillengine.model_registry import (
    ThinkingLevel,
    Transport,
    map_thinking_level_to_openai_effort,
)


class OpenAIFunction(TypedDict):
    """OpenAI function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class OpenAITool(TypedDict):
    """OpenAI tool definition."""

    type: str
    function: OpenAIFunction


class OpenAIMessage(TypedDict, total=False):
    """OpenAI message format."""

    role: str
    content: str | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


class OpenAIAdapter(LLMAdapter):
    """
    OpenAI adapter for the skills engine.

    Example:
        from openai import AsyncOpenAI
        from skillengine import SkillsEngine
        from skillengine.adapters import OpenAIAdapter

        engine = SkillsEngine(config=...)
        client = AsyncOpenAI()
        adapter = OpenAIAdapter(engine, client)

        response = await adapter.chat([
            Message(role="user", content="List my GitHub PRs")
        ])
    """

    def __init__(
        self,
        engine: SkillsEngine,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4-turbo-preview",
        enable_tools: bool = True,
        transport: Transport = "sse",
        session_id: str | None = None,
    ) -> None:
        super().__init__(engine)
        self.client = client or AsyncOpenAI()
        self.model = model
        self.enable_tools = enable_tools
        self.transport = transport
        self.session_id = session_id

    def _get_openai_tools(self) -> list[OpenAITool]:
        """Convert tool definitions to OpenAI format."""
        tool_defs = self.get_tool_definitions()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tool_defs
        ]

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if the model is a reasoning model (o3/o4-mini pattern)."""
        import re

        return bool(re.search(r"\bo[34]", model.lower()))

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AgentResponse:
        """Send a chat request to OpenAI."""
        # Build system prompt with skills
        full_system = self.build_system_prompt(system_prompt or "")

        # Format messages for OpenAI
        openai_messages: list[dict[str, Any]] = []

        if full_system:
            openai_messages.append(
                {
                    "role": "system",
                    "content": full_system,
                }
            )

        for msg in messages:
            openai_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        # Add tools if enabled
        if self.enable_tools:
            tools = self._get_openai_tools()
            if tools:
                request_kwargs["tools"] = tools

        # Add reasoning effort for reasoning models
        level = thinking_level or "off"
        if level != "off" and self._is_reasoning_model(self.model):
            request_kwargs["reasoning_effort"] = map_thinking_level_to_openai_effort(level)

        # Call OpenAI
        response = await self.client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract tool calls if any
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def _build_openai_messages(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with system prompt."""
        full_system = self.build_system_prompt(system_prompt or "")
        openai_messages: list[dict[str, Any]] = []
        if full_system:
            openai_messages.append({"role": "system", "content": full_system})
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})
        return openai_messages

    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response from OpenAI (text deltas only)."""
        async for event in self.chat_stream_events(
            messages, system_prompt, thinking_level=thinking_level
        ):
            if event.type == "text_delta":
                yield event.content

    async def chat_stream_events(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream structured events from OpenAI.

        Maps OpenAI streaming chunks to StreamEvent types:
        - Content deltas → text_start/text_delta/text_end
        - Tool call deltas → tool_call_start/tool_call_delta/tool_call_end
        """
        openai_messages = self._build_openai_messages(messages, system_prompt)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }

        if self.enable_tools:
            tools = self._get_openai_tools()
            if tools:
                request_kwargs["tools"] = tools

        # Add reasoning effort for reasoning models
        level = thinking_level or "off"
        if level != "off" and self._is_reasoning_model(self.model):
            request_kwargs["reasoning_effort"] = map_thinking_level_to_openai_effort(level)

        stream = await self.client.chat.completions.create(**request_kwargs)

        text_started = False
        # Track active tool calls: index -> {id, name, args_buffer}
        active_tool_calls: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # --- Text content ---
            if delta.content:
                if not text_started:
                    yield StreamEvent(type="text_start")
                    text_started = True
                yield StreamEvent(type="text_delta", content=delta.content)

            # --- Tool calls ---
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in active_tool_calls:
                        # New tool call starting
                        tc_id = tc_delta.id or ""
                        tc_name = (
                            tc_delta.function.name
                            if tc_delta.function and tc_delta.function.name
                            else ""
                        )
                        active_tool_calls[idx] = {
                            "id": tc_id,
                            "name": tc_name,
                            "args": "",
                        }
                        # End text if it was streaming
                        if text_started:
                            yield StreamEvent(type="text_end")
                            text_started = False
                        yield StreamEvent(
                            type="tool_call_start",
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                        )

                    # Accumulate arguments
                    if tc_delta.function and tc_delta.function.arguments:
                        active_tool_calls[idx]["args"] += tc_delta.function.arguments
                        yield StreamEvent(
                            type="tool_call_delta",
                            tool_call_id=active_tool_calls[idx]["id"],
                            tool_name=active_tool_calls[idx]["name"],
                            args_delta=tc_delta.function.arguments,
                        )

            # --- Finish ---
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                if text_started:
                    yield StreamEvent(type="text_end")
                    text_started = False

                # Close any open tool calls
                for idx, tc_info in active_tool_calls.items():
                    yield StreamEvent(
                        type="tool_call_end",
                        tool_call_id=tc_info["id"],
                        tool_name=tc_info["name"],
                    )

                yield StreamEvent(type="done", finish_reason=finish_reason)
