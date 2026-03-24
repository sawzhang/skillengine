"""
Base LLM adapter interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from skillengine.engine import SkillsEngine
from skillengine.events import StreamEvent
from skillengine.extensions.models import ToolInfo
from skillengine.model_registry import ThinkingLevel, TokenUsage
from skillengine.models import SkillSnapshot


class ToolParameter(TypedDict, total=False):
    """Tool parameter definition."""

    type: str
    description: str
    enum: list[str]


class ToolProperties(TypedDict):
    """Tool properties schema."""

    type: str
    properties: dict[str, ToolParameter]
    required: list[str]


class ToolDefinition(TypedDict):
    """Standard tool definition format."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    token_usage: TokenUsage | None = None


class LLMAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.

    Adapters integrate the skills engine with specific LLM providers,
    handling system prompt injection, tool execution, and response parsing.

    Example implementation for a custom provider:

        class MyLLMAdapter(LLMAdapter):
            def __init__(self, engine: SkillsEngine, client: MyLLMClient):
                super().__init__(engine)
                self.client = client

            async def chat(self, messages: list[Message]) -> AgentResponse:
                # Inject skills into system prompt
                system_prompt = self.build_system_prompt()

                # Call LLM
                response = await self.client.chat(
                    system=system_prompt,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )

                return AgentResponse(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
    """

    def __init__(self, engine: SkillsEngine) -> None:
        self.engine = engine

    def get_snapshot(self) -> SkillSnapshot:
        """Get the current skills snapshot."""
        return self.engine.get_snapshot()

    def build_system_prompt(self, base_prompt: str = "") -> str:
        """
        Build a system prompt with skills injected.

        Args:
            base_prompt: Base system prompt to extend

        Returns:
            System prompt with skills appended
        """
        snapshot = self.get_snapshot()
        skills_prompt = snapshot.prompt

        if not skills_prompt:
            return base_prompt

        if base_prompt:
            return f"{base_prompt}\n\n{skills_prompt}"
        return skills_prompt

    def get_tool_definitions(
        self, extra_tools: list[ToolInfo] | None = None
    ) -> list[ToolDefinition]:
        """
        Get standard tool definitions for LLM function calling.

        Returns a list of tools that can be used with OpenAI, Anthropic, or
        other LLM providers that support function calling.

        Args:
            extra_tools: Additional tools from extensions to include

        Returns:
            List of tool definitions
        """
        tools: list[ToolDefinition] = [
            {
                "name": "execute",
                "description": "Execute a single shell command and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "execute_script",
                "description": "Execute a multi-line shell script and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "The shell script content to execute",
                        },
                    },
                    "required": ["script"],
                },
            },
        ]
        if extra_tools:
            for t in extra_tools:
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                )
        return tools

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AgentResponse:
        """
        Send a chat request to the LLM.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt (skills will be appended)
            thinking_level: Optional thinking budget level

        Returns:
            AgentResponse with LLM output
        """
        pass

    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response from the LLM.

        Default implementation falls back to non-streaming chat.
        Override for true streaming support.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            thinking_level: Optional thinking budget level

        Yields:
            Response chunks
        """
        response = await self.chat(messages, system_prompt, thinking_level=thinking_level)
        yield response.content

    async def chat_stream_events(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream structured events from the LLM.

        Default implementation wraps ``chat()`` into a sequence of events:
        text_start → text_delta → text_end → done.

        Override in provider-specific adapters for true granular streaming
        (separate thinking, text, and tool_call deltas).

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt

        Yields:
            StreamEvent objects
        """
        response = await self.chat(messages, system_prompt, thinking_level=thinking_level)

        # Emit text
        if response.content:
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content=response.content)
            yield StreamEvent(type="text_end")

        # Emit tool calls (as complete, non-streamed events)
        for tc in response.tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", "")
            if isinstance(tc_args, dict):
                import json

                tc_args = json.dumps(tc_args)
            yield StreamEvent(
                type="tool_call_start",
                tool_call_id=tc_id,
                tool_name=tc_name,
            )
            yield StreamEvent(
                type="tool_call_delta",
                tool_call_id=tc_id,
                tool_name=tc_name,
                args_delta=tc_args,
            )
            yield StreamEvent(
                type="tool_call_end",
                tool_call_id=tc_id,
                tool_name=tc_name,
            )

        yield StreamEvent(type="done", finish_reason="complete")

    async def run_agent_loop(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_turns: int = 10,
    ) -> list[Message]:
        """
        Run an agent loop with tool execution.

        Continues until the LLM stops requesting tools or max_turns is reached.

        Args:
            messages: Initial messages
            system_prompt: System prompt
            max_turns: Maximum number of turns

        Returns:
            Complete conversation including tool results
        """
        conversation = list(messages)

        for _ in range(max_turns):
            response = await self.chat(conversation, system_prompt)

            # Add assistant response
            conversation.append(
                Message(
                    role="assistant",
                    content=response.content,
                )
            )

            # Check for tool calls
            if not response.tool_calls:
                break

            # Execute tools and add results
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                conversation.append(
                    Message(
                        role="tool",
                        content=result,
                        metadata={"tool_call_id": tool_call.get("id")},
                    )
                )

        return conversation

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call."""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        if name == "bash" or name == "execute":
            command = args.get("command", "")
            result = await self.engine.execute(command)
            if result.success:
                return result.output
            return f"Error: {result.error}"

        if name == "execute_script":
            script = args.get("script", "")
            result = await self.engine.execute_script(script)
            if result.success:
                return result.output
            return f"Error: {result.error}"

        return f"Unknown tool: {name}"
