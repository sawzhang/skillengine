"""
Anthropic adapter for the skills engine.

Requires the 'anthropic' extra: pip install skillengine[anthropic]
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

try:
    from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "Anthropic adapter requires the 'anthropic' package. "
        "Install with: pip install skillengine[anthropic]"
    )

from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
from skillengine.engine import SkillsEngine
from skillengine.events import StreamEvent
from skillengine.model_registry import (
    ThinkingLevel,
    TokenUsage,
    adjust_max_tokens_for_thinking,
    map_thinking_level_to_anthropic_effort,
    supports_adaptive_thinking,
)


class AnthropicInputSchema(TypedDict):
    """Anthropic tool input schema."""

    type: str
    properties: dict[str, Any]
    required: list[str]


class AnthropicTool(TypedDict):
    """Anthropic tool definition."""

    name: str
    description: str
    input_schema: AnthropicInputSchema


class AnthropicMessage(TypedDict, total=False):
    """Anthropic message format."""

    role: str
    content: str | list[dict[str, Any]]


class AnthropicAdapter(LLMAdapter):
    """
    Anthropic adapter for the skills engine.

    Example:
        from anthropic import AsyncAnthropic
        from skillengine import SkillsEngine
        from skillengine.adapters import AnthropicAdapter

        engine = SkillsEngine(config=...)
        client = AsyncAnthropic()
        adapter = AnthropicAdapter(engine, client)

        response = await adapter.chat([
            Message(role="user", content="List my GitHub PRs")
        ])
    """

    def __init__(
        self,
        engine: SkillsEngine,
        client: AsyncAnthropic | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        enable_tools: bool = True,
    ) -> None:
        super().__init__(engine)
        self.client = client or AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.enable_tools = enable_tools

    def _get_anthropic_tools(self) -> list[AnthropicTool]:
        """Convert tool definitions to Anthropic format."""
        tool_defs = self.get_tool_definitions()
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": tool["parameters"]["type"],
                    "properties": tool["parameters"]["properties"],
                    "required": tool["parameters"]["required"],
                },
            }
            for tool in tool_defs
        ]

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AgentResponse:
        """Send a chat request to Anthropic."""
        # Build system prompt with skills
        full_system = self.build_system_prompt(system_prompt or "")

        # Format messages for Anthropic
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic handles system separately
                continue
            anthropic_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }

        if full_system:
            request_kwargs["system"] = full_system

        # Add tools if enabled
        if self.enable_tools:
            tools = self._get_anthropic_tools()
            if tools:
                request_kwargs["tools"] = tools

        # Add thinking configuration
        level = thinking_level or "off"
        if level != "off":
            if supports_adaptive_thinking(self.model):
                effort = map_thinking_level_to_anthropic_effort(level)
                request_kwargs["thinking"] = {"type": "adaptive"}
                request_kwargs["output_config"] = {"effort": effort}
            else:
                max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
                    self.max_tokens, 128_000, level
                )
                request_kwargs["max_tokens"] = max_tokens
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }

        # Call Anthropic
        response = await self.client.messages.create(**request_kwargs)

        # Extract content
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        # Extract thinking tokens from usage if available
        thinking_tokens = 0
        if hasattr(response.usage, "thinking_tokens"):
            thinking_tokens = response.usage.thinking_tokens or 0

        token_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            thinking_tokens=thinking_tokens,
        )

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            token_usage=token_usage,
        )

    def _build_anthropic_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Build Anthropic-format messages (excluding system role)."""
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            anthropic_messages.append({"role": msg.role, "content": msg.content})
        return anthropic_messages

    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response from Anthropic (text deltas only)."""
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
        Stream structured events from Anthropic.

        Maps Anthropic streaming events to StreamEvent types:
        - content_block_start (text) → text_start
        - content_block_delta (text_delta) → text_delta
        - content_block_stop (text) → text_end
        - content_block_start (thinking) → thinking_start
        - content_block_delta (thinking_delta) → thinking_delta
        - content_block_stop (thinking) → thinking_end
        - content_block_start (tool_use) → tool_call_start
        - content_block_delta (input_json_delta) → tool_call_delta
        - content_block_stop (tool_use) → tool_call_end
        """

        full_system = self.build_system_prompt(system_prompt or "")
        anthropic_messages = self._build_anthropic_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if full_system:
            request_kwargs["system"] = full_system
        if self.enable_tools:
            tools = self._get_anthropic_tools()
            if tools:
                request_kwargs["tools"] = tools

        # Add thinking configuration
        level = thinking_level or "off"
        if level != "off":
            if supports_adaptive_thinking(self.model):
                effort = map_thinking_level_to_anthropic_effort(level)
                request_kwargs["thinking"] = {"type": "adaptive"}
                request_kwargs["output_config"] = {"effort": effort}
            else:
                max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
                    self.max_tokens, 128_000, level
                )
                request_kwargs["max_tokens"] = max_tokens
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }

        # Track current block type for mapping stop events
        # block_index -> {"type": "text"|"thinking"|"tool_use", "id": ..., "name": ...}
        active_blocks: dict[int, dict[str, str]] = {}

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    idx = event.index
                    if block.type == "text":
                        active_blocks[idx] = {"type": "text"}
                        yield StreamEvent(type="text_start")
                    elif block.type == "thinking":
                        active_blocks[idx] = {"type": "thinking"}
                        yield StreamEvent(type="thinking_start")
                    elif block.type == "tool_use":
                        tc_id = block.id
                        tc_name = block.name
                        active_blocks[idx] = {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                        }
                        yield StreamEvent(
                            type="tool_call_start",
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                        )

                elif event_type == "content_block_delta":
                    idx = event.index
                    delta = event.delta
                    block_info = active_blocks.get(idx, {})

                    if delta.type == "text_delta":
                        yield StreamEvent(type="text_delta", content=delta.text)
                    elif delta.type == "thinking_delta":
                        yield StreamEvent(
                            type="thinking_delta",
                            content=delta.thinking,
                        )
                    elif delta.type == "input_json_delta":
                        yield StreamEvent(
                            type="tool_call_delta",
                            tool_call_id=block_info.get("id"),
                            tool_name=block_info.get("name"),
                            args_delta=delta.partial_json,
                        )

                elif event_type == "content_block_stop":
                    idx = event.index
                    block_info = active_blocks.pop(idx, {})
                    btype = block_info.get("type", "")

                    if btype == "text":
                        yield StreamEvent(type="text_end")
                    elif btype == "thinking":
                        yield StreamEvent(type="thinking_end")
                    elif btype == "tool_use":
                        yield StreamEvent(
                            type="tool_call_end",
                            tool_call_id=block_info.get("id"),
                            tool_name=block_info.get("name"),
                        )

                elif event_type == "message_stop":
                    yield StreamEvent(type="done", finish_reason="complete")
