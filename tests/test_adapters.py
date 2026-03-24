"""Tests for LLM adapters."""

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from skillengine import SkillsConfig, SkillsEngine
from skillengine.adapters.base import AgentResponse, LLMAdapter, Message


class MockAdapter(LLMAdapter):
    """Mock adapter for testing base class functionality."""

    def __init__(self, engine: SkillsEngine, responses: list[AgentResponse] | None = None):
        super().__init__(engine)
        self.responses = responses or []
        self.response_index = 0
        self.chat_calls: list[tuple[list[Message], str | None]] = []

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Record call and return mock response."""
        self.chat_calls.append((messages, system_prompt))

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response

        return AgentResponse(content="Mock response", tool_calls=[])


class TestLLMAdapterBase:
    """Tests for LLMAdapter base class."""

    def test_get_snapshot(self, skill_dir: Path) -> None:
        """Should get snapshot from engine."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        snapshot = adapter.get_snapshot()

        assert snapshot is not None
        assert len(snapshot.skills) > 0

    def test_build_system_prompt_empty_base(self, skill_dir: Path) -> None:
        """Should build prompt from skills when base is empty."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        prompt = adapter.build_system_prompt("")

        assert "<skills>" in prompt
        assert "simple-skill" in prompt

    def test_build_system_prompt_with_base(self, skill_dir: Path) -> None:
        """Should append skills to base prompt."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        prompt = adapter.build_system_prompt("You are a helpful assistant.")

        assert prompt.startswith("You are a helpful assistant.")
        assert "<skills>" in prompt

    def test_build_system_prompt_no_skills(self, tmp_path: Path) -> None:
        """Should return base prompt when no skills."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[empty_dir]))
        adapter = MockAdapter(engine)

        prompt = adapter.build_system_prompt("Base prompt")

        assert prompt == "Base prompt"


@pytest.mark.asyncio
class TestLLMAdapterChat:
    """Async tests for LLMAdapter chat methods."""

    async def test_chat_stream_default(self, skill_dir: Path) -> None:
        """Default chat_stream should fall back to chat."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(
            engine,
            responses=[AgentResponse(content="Hello!", tool_calls=[])],
        )

        chunks = []
        async for chunk in adapter.chat_stream([Message(role="user", content="Hi")]):
            chunks.append(chunk)

        assert chunks == ["Hello!"]

    async def test_run_agent_loop_no_tools(self, skill_dir: Path) -> None:
        """Agent loop should return when no tool calls."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(
            engine,
            responses=[AgentResponse(content="Done!", tool_calls=[])],
        )

        conversation = await adapter.run_agent_loop(
            [Message(role="user", content="Hello")],
            max_turns=5,
        )

        assert len(conversation) == 2  # User + Assistant
        assert conversation[0].role == "user"
        assert conversation[1].role == "assistant"
        assert conversation[1].content == "Done!"

    async def test_run_agent_loop_with_tool_calls(self, skill_dir: Path) -> None:
        """Agent loop should execute tool calls."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(
            engine,
            responses=[
                AgentResponse(
                    content="Let me execute that.",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "execute",
                        "arguments": {"command": "echo 'hello'"},
                    }],
                ),
                AgentResponse(content="Done!", tool_calls=[]),
            ],
        )

        conversation = await adapter.run_agent_loop(
            [Message(role="user", content="Run echo")],
            max_turns=5,
        )

        # User + Assistant (with tool) + Tool result + Assistant (final)
        assert len(conversation) == 4
        assert conversation[0].role == "user"
        assert conversation[1].role == "assistant"
        assert conversation[2].role == "tool"
        assert "hello" in conversation[2].content
        assert conversation[3].role == "assistant"
        assert conversation[3].content == "Done!"

    async def test_run_agent_loop_max_turns(self, skill_dir: Path) -> None:
        """Agent loop should stop at max_turns."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        # Return tool calls forever
        adapter = MockAdapter(
            engine,
            responses=[
                AgentResponse(
                    content="Again",
                    tool_calls=[{
                        "id": f"call_{i}",
                        "name": "execute",
                        "arguments": {"command": "echo 'loop'"},
                    }],
                )
                for i in range(10)
            ],
        )

        conversation = await adapter.run_agent_loop(
            [Message(role="user", content="Loop")],
            max_turns=3,
        )

        # Should stop after 3 turns (3 assistant + 3 tool = 6, plus initial user = 7)
        assert len(adapter.chat_calls) == 3


@pytest.mark.asyncio
class TestToolExecution:
    """Tests for tool execution in adapters."""

    async def test_execute_tool_bash(self, skill_dir: Path) -> None:
        """Should execute bash command."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        result = await adapter._execute_tool({
            "name": "bash",
            "arguments": {"command": "echo 'test'"},
        })

        assert "test" in result

    async def test_execute_tool_execute(self, skill_dir: Path) -> None:
        """Should execute 'execute' tool."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        result = await adapter._execute_tool({
            "name": "execute",
            "arguments": {"command": "pwd"},
        })

        # Should return current working directory
        assert "/" in result

    async def test_execute_tool_unknown(self, skill_dir: Path) -> None:
        """Should return error for unknown tool."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        result = await adapter._execute_tool({
            "name": "unknown_tool",
            "arguments": {},
        })

        assert "Unknown tool" in result

    async def test_execute_tool_error(self, skill_dir: Path) -> None:
        """Should return error message on command failure."""
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
        adapter = MockAdapter(engine)

        result = await adapter._execute_tool({
            "name": "execute",
            "arguments": {"command": "exit 1"},
        })

        assert "Error" in result


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_basic(self) -> None:
        """Should create message with basic fields."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self) -> None:
        """Should create message with metadata."""
        msg = Message(
            role="tool",
            content="Result",
            metadata={"tool_call_id": "call_123"},
        )

        assert msg.metadata["tool_call_id"] == "call_123"


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_response_basic(self) -> None:
        """Should create response with basic fields."""
        response = AgentResponse(content="Hello")

        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.finish_reason is None
        assert response.usage == {}

    def test_response_with_tool_calls(self) -> None:
        """Should create response with tool calls."""
        response = AgentResponse(
            content="Let me check",
            tool_calls=[
                {"id": "call_1", "name": "search", "arguments": {"query": "test"}},
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search"
        assert response.finish_reason == "tool_calls"
        assert response.usage["prompt_tokens"] == 100
