"""Tests for structured stream events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.events import (
    BEFORE_TOOL_CALL,
    EventBus,
    StreamEvent,
    ToolCallEventResult,
)


# ---------------------------------------------------------------------------
# StreamEvent dataclass tests
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_text_delta(self) -> None:
        event = StreamEvent(type="text_delta", content="hello")
        assert event.type == "text_delta"
        assert event.content == "hello"
        assert event.tool_name is None
        assert event.tool_call_id is None

    def test_tool_call_start(self) -> None:
        event = StreamEvent(
            type="tool_call_start", tool_call_id="tc1", tool_name="execute"
        )
        assert event.tool_name == "execute"
        assert event.tool_call_id == "tc1"

    def test_tool_call_delta(self) -> None:
        event = StreamEvent(
            type="tool_call_delta",
            tool_call_id="tc1",
            tool_name="execute",
            args_delta='{"command":',
        )
        assert event.args_delta == '{"command":'

    def test_done(self) -> None:
        event = StreamEvent(type="done", finish_reason="complete")
        assert event.finish_reason == "complete"

    def test_error(self) -> None:
        event = StreamEvent(type="error", error="Connection failed")
        assert event.error == "Connection failed"

    def test_defaults(self) -> None:
        event = StreamEvent(type="text_start")
        assert event.content == ""
        assert event.tool_name is None
        assert event.tool_call_id is None
        assert event.turn == 0
        assert event.error is None
        assert event.finish_reason is None
        assert event.args_delta is None

    def test_turn_start(self) -> None:
        event = StreamEvent(type="turn_start", turn=3)
        assert event.turn == 3

    def test_tool_result(self) -> None:
        event = StreamEvent(
            type="tool_result",
            tool_call_id="tc1",
            tool_name="execute",
            content="file1.txt\nfile2.txt",
            turn=0,
        )
        assert event.content == "file1.txt\nfile2.txt"

    def test_thinking_delta(self) -> None:
        event = StreamEvent(type="thinking_delta", content="Let me think...")
        assert event.type == "thinking_delta"
        assert event.content == "Let me think..."


# ---------------------------------------------------------------------------
# LLMAdapter base chat_stream_events tests
# ---------------------------------------------------------------------------


class TestLLMAdapterStreamEvents:
    """Test the default chat_stream_events implementation in LLMAdapter base."""

    def test_default_wraps_chat(self) -> None:
        """Default implementation should wrap chat() into events."""
        from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        class MockAdapter(LLMAdapter):
            async def chat(self, messages, system_prompt=None, **kwargs):
                return AgentResponse(content="Hello!", tool_calls=[])

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        adapter = MockAdapter(engine)

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(
                adapter.chat_stream_events([Message(role="user", content="Hi")])
            )
        )

        types = [e.type for e in events]
        assert types == ["text_start", "text_delta", "text_end", "done"]
        assert events[1].content == "Hello!"
        assert events[3].finish_reason == "complete"

    def test_default_with_tool_calls(self) -> None:
        """Default implementation should emit tool_call events for tool_calls."""
        from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        class MockAdapter(LLMAdapter):
            async def chat(self, messages, system_prompt=None, **kwargs):
                return AgentResponse(
                    content="Let me check.",
                    tool_calls=[
                        {"id": "tc1", "name": "execute", "arguments": '{"command":"ls"}'},
                    ],
                )

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        adapter = MockAdapter(engine)

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(
                adapter.chat_stream_events([Message(role="user", content="list")])
            )
        )

        types = [e.type for e in events]
        assert "text_start" in types
        assert "text_delta" in types
        assert "text_end" in types
        assert "tool_call_start" in types
        assert "tool_call_delta" in types
        assert "tool_call_end" in types
        assert "done" in types

        tc_start = next(e for e in events if e.type == "tool_call_start")
        assert tc_start.tool_name == "execute"
        assert tc_start.tool_call_id == "tc1"

    def test_default_empty_content(self) -> None:
        """No text events when content is empty (only tool calls)."""
        from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        class MockAdapter(LLMAdapter):
            async def chat(self, messages, system_prompt=None, **kwargs):
                return AgentResponse(
                    content="",
                    tool_calls=[
                        {"id": "tc1", "name": "execute", "arguments": '{"command":"ls"}'},
                    ],
                )

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        adapter = MockAdapter(engine)

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(
                adapter.chat_stream_events([Message(role="user", content="list")])
            )
        )

        types = [e.type for e in events]
        assert "text_start" not in types
        assert "text_delta" not in types
        assert "text_end" not in types
        assert "tool_call_start" in types

    def test_chat_stream_wraps_events(self) -> None:
        """chat_stream() should yield only text content from events."""
        from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        class MockAdapter(LLMAdapter):
            async def chat(self, messages, system_prompt=None, **kwargs):
                return AgentResponse(content="Hello world!", tool_calls=[])

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        adapter = MockAdapter(engine)

        chunks = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(
                adapter.chat_stream([Message(role="user", content="Hi")])
            )
        )
        assert chunks == ["Hello world!"]


# ---------------------------------------------------------------------------
# AgentRunner.chat_stream_events() tests
# ---------------------------------------------------------------------------


class TestAgentRunnerStreamEvents:
    """Test AgentRunner.chat_stream_events() with the full agent loop."""

    def _make_runner(self, events: EventBus | None = None) -> Any:
        """Create a minimal AgentRunner with mocked LLM."""
        from skillengine.agent import AgentConfig, AgentRunner
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        agent_config = AgentConfig(
            model="test-model",
            system_prompt="test",
            enable_tools=True,
            auto_execute=True,
        )
        return AgentRunner(engine, agent_config, events=events)

    def test_simple_text_response(self) -> None:
        """Simple text response should emit text_start, text_delta, text_end, done."""
        runner = self._make_runner()

        # Mock _call_llm_stream to yield a simple text response
        async def mock_stream(messages):
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content="Hello ")
            yield StreamEvent(type="text_delta", content="world!")
            yield StreamEvent(type="text_end")
            yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("hi"))
        )

        types = [e.type for e in events]
        assert types == [
            "turn_start",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done",  # from _call_llm_stream
            "turn_end",
            "done",  # from agent loop completion
        ]

        # Text content should be collected
        text_deltas = [e.content for e in events if e.type == "text_delta"]
        assert text_deltas == ["Hello ", "world!"]

    def test_thinking_events(self) -> None:
        """Thinking events should pass through from LLM stream."""
        runner = self._make_runner()

        async def mock_stream(messages):
            yield StreamEvent(type="thinking_start")
            yield StreamEvent(type="thinking_delta", content="Let me think...")
            yield StreamEvent(type="thinking_end")
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content="The answer is 42.")
            yield StreamEvent(type="text_end")
            yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("think"))
        )

        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        assert "text_delta" in types

        thinking = [e.content for e in events if e.type == "thinking_delta"]
        assert thinking == ["Let me think..."]

    def test_tool_call_and_result(self) -> None:
        """Should emit tool call events, then tool_result, then continue."""
        runner = self._make_runner()

        call_count = 0

        async def mock_stream(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call
                yield StreamEvent(type="tool_call_start", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(
                    type="tool_call_delta", tool_call_id="tc1",
                    tool_name="execute", args_delta='{"command":"ls"}',
                )
                yield StreamEvent(type="tool_call_end", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(type="done", finish_reason="tool_calls")
            else:
                # Second call: text response
                yield StreamEvent(type="text_start")
                yield StreamEvent(type="text_delta", content="Here are the files.")
                yield StreamEvent(type="text_end")
                yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        # Mock tool execution
        async def mock_execute(command, cwd=None, **kwargs):
            result = MagicMock()
            result.success = True
            result.output = "file1.txt\nfile2.txt"
            return result

        runner.engine.execute = mock_execute

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("list files"))
        )

        types = [e.type for e in events]

        # Should have: turn_start, tool_call_*, done, turn_end, tool_result, turn_start, text_*, done, turn_end, done
        assert "tool_call_start" in types
        assert "tool_call_delta" in types
        assert "tool_call_end" in types
        assert "tool_result" in types
        assert types.count("turn_start") == 2
        assert types.count("turn_end") == 2

        # Check tool_result content
        tool_results = [e for e in events if e.type == "tool_result"]
        assert len(tool_results) == 1
        assert "file1.txt" in tool_results[0].content
        assert tool_results[0].tool_name == "execute"

    def test_tool_blocked_emits_tool_result(self) -> None:
        """Blocked tool call should emit tool_result with blocked message."""
        bus = EventBus()

        def block_all(event):
            return ToolCallEventResult(block=True, reason="No execution allowed")

        bus.on(BEFORE_TOOL_CALL, block_all)

        runner = self._make_runner(events=bus)
        call_count = 0

        async def mock_stream(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamEvent(type="tool_call_start", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(
                    type="tool_call_delta", tool_call_id="tc1",
                    tool_name="execute", args_delta='{"command":"rm -rf /"}',
                )
                yield StreamEvent(type="tool_call_end", tool_call_id="tc1", tool_name="execute")
                yield StreamEvent(type="done", finish_reason="tool_calls")
            else:
                yield StreamEvent(type="text_start")
                yield StreamEvent(type="text_delta", content="Understood.")
                yield StreamEvent(type="text_end")
                yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("do it"))
        )

        tool_results = [e for e in events if e.type == "tool_result"]
        assert len(tool_results) == 1
        assert "[Blocked]" in tool_results[0].content
        assert "No execution allowed" in tool_results[0].content

    def test_chat_stream_wraps_events(self) -> None:
        """chat_stream() should yield only text_delta content."""
        runner = self._make_runner()

        async def mock_stream(messages):
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content="Hello")
            yield StreamEvent(type="text_delta", content=" world")
            yield StreamEvent(type="text_end")
            yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        chunks = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream("test"))
        )

        assert chunks == ["Hello", " world"]

    def test_turn_numbers_on_events(self) -> None:
        """All events within a turn should have the correct turn number."""
        runner = self._make_runner()

        async def mock_stream(messages):
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content="Done")
            yield StreamEvent(type="text_end")
            yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("test"))
        )

        # All events from the LLM stream should have turn=0
        llm_events = [e for e in events if e.type in ("text_start", "text_delta", "text_end")]
        for e in llm_events:
            assert e.turn == 0

    def test_error_event(self) -> None:
        """Error during streaming should emit error event."""
        runner = self._make_runner()

        async def mock_stream(messages):
            yield StreamEvent(type="text_start")
            raise RuntimeError("LLM connection lost")

        runner._call_llm_stream = mock_stream

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="LLM connection lost"):
            asyncio.get_event_loop().run_until_complete(
                _collect_async_iter_safe(runner.chat_stream_events("test"), events)
            )

        types = [e.type for e in events]
        assert "error" in types
        error_event = next(e for e in events if e.type == "error")
        assert "LLM connection lost" in error_event.error

    def test_max_turns_emits_done(self) -> None:
        """Reaching max_turns should emit done with finish_reason='max_turns'."""
        from skillengine.agent import AgentConfig, AgentRunner
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        agent_config = AgentConfig(
            model="test-model",
            system_prompt="test",
            enable_tools=True,
            auto_execute=True,
            max_turns=1,  # Only 1 turn allowed
        )
        runner = AgentRunner(engine, agent_config)

        async def mock_stream(messages):
            yield StreamEvent(type="tool_call_start", tool_call_id="tc1", tool_name="execute")
            yield StreamEvent(
                type="tool_call_delta", tool_call_id="tc1",
                tool_name="execute", args_delta='{"command":"ls"}',
            )
            yield StreamEvent(type="tool_call_end", tool_call_id="tc1", tool_name="execute")
            yield StreamEvent(type="done", finish_reason="tool_calls")

        runner._call_llm_stream = mock_stream

        async def mock_execute(command, cwd=None, **kwargs):
            result = MagicMock()
            result.success = True
            result.output = "ok"
            return result

        runner.engine.execute = mock_execute

        events = asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("list"))
        )

        done_events = [e for e in events if e.type == "done"]
        # Last done should be max_turns
        assert done_events[-1].finish_reason == "max_turns"

    def test_conversation_updated(self) -> None:
        """Conversation should be updated after streaming."""
        runner = self._make_runner()

        async def mock_stream(messages):
            yield StreamEvent(type="text_start")
            yield StreamEvent(type="text_delta", content="Response text")
            yield StreamEvent(type="text_end")
            yield StreamEvent(type="done", finish_reason="stop")

        runner._call_llm_stream = mock_stream

        asyncio.get_event_loop().run_until_complete(
            _collect_async_iter(runner.chat_stream_events("hello"))
        )

        history = runner.get_history()
        assert len(history) == 2  # user + assistant
        assert history[0].role == "user"
        assert history[0].content == "hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Response text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_async_iter(ait: AsyncIterator) -> list:
    """Collect all items from an async iterator."""
    items = []
    async for item in ait:
        items.append(item)
    return items


async def _collect_async_iter_safe(ait: AsyncIterator, out: list) -> None:
    """Collect items from async iterator, appending to out. Lets exceptions propagate."""
    async for item in ait:
        out.append(item)
