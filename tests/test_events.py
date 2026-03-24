"""Tests for the event system."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.events import (
    AGENT_END,
    AGENT_START,
    AFTER_TOOL_RESULT,
    BEFORE_TOOL_CALL,
    CONTEXT_TRANSFORM,
    INPUT,
    TURN_END,
    TURN_START,
    AfterToolResultEvent,
    AgentEndEvent,
    AgentStartEvent,
    BeforeToolCallEvent,
    ContextTransformEvent,
    ContextTransformEventResult,
    EventBus,
    InputEvent,
    InputEventResult,
    ToolCallEventResult,
    ToolResultEventResult,
    TurnEndEvent,
    TurnStartEvent,
)


# ---------------------------------------------------------------------------
# EventBus core tests
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_on_and_emit(self) -> None:
        bus = EventBus()
        received: list[str] = []

        bus.on("test", lambda data: received.append(data))

        asyncio.get_event_loop().run_until_complete(bus.emit("test", "hello"))
        assert received == ["hello"]

    def test_on_decorator(self) -> None:
        bus = EventBus()
        received: list[str] = []

        @bus.on("test")
        def handler(data: str) -> None:
            received.append(data)

        asyncio.get_event_loop().run_until_complete(bus.emit("test", "world"))
        assert received == ["world"]

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[str] = []

        unsub = bus.on("test", lambda data: received.append(data))
        asyncio.get_event_loop().run_until_complete(bus.emit("test", "a"))
        assert received == ["a"]

        unsub()
        asyncio.get_event_loop().run_until_complete(bus.emit("test", "b"))
        assert received == ["a"]  # handler removed, no "b"

    def test_off(self) -> None:
        bus = EventBus()
        received: list[str] = []

        def handler(data: str) -> None:
            received.append(data)

        bus.on("test", handler)
        bus.off("test", handler)

        asyncio.get_event_loop().run_until_complete(bus.emit("test", "x"))
        assert received == []

    def test_off_by_source(self) -> None:
        bus = EventBus()
        received: list[str] = []

        bus.on("test", lambda d: received.append("ext1"), source="ext1")
        bus.on("test", lambda d: received.append("ext2"), source="ext2")

        removed = bus.off_by_source("ext1")
        assert removed == 1

        asyncio.get_event_loop().run_until_complete(bus.emit("test", None))
        assert received == ["ext2"]

    def test_clear_specific_event(self) -> None:
        bus = EventBus()
        bus.on("a", lambda d: None)
        bus.on("b", lambda d: None)

        bus.clear("a")
        assert bus.handler_count == 1
        assert not bus.has_handlers("a")
        assert bus.has_handlers("b")

    def test_clear_all(self) -> None:
        bus = EventBus()
        bus.on("a", lambda d: None)
        bus.on("b", lambda d: None)

        bus.clear()
        assert bus.handler_count == 0

    def test_priority_order(self) -> None:
        bus = EventBus()
        order: list[int] = []

        bus.on("test", lambda d: order.append(3), priority=30)
        bus.on("test", lambda d: order.append(1), priority=10)
        bus.on("test", lambda d: order.append(2), priority=20)

        asyncio.get_event_loop().run_until_complete(bus.emit("test", None))
        assert order == [1, 2, 3]

    def test_async_handler(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def async_handler(data: str) -> None:
            received.append(data)

        bus.on("test", async_handler)

        asyncio.get_event_loop().run_until_complete(bus.emit("test", "async"))
        assert received == ["async"]

    def test_handler_return_values(self) -> None:
        bus = EventBus()

        bus.on("test", lambda d: "result_a", priority=1)
        bus.on("test", lambda d: None, priority=2)  # None is filtered out
        bus.on("test", lambda d: "result_b", priority=3)

        results = asyncio.get_event_loop().run_until_complete(bus.emit("test", None))
        assert results == ["result_a", "result_b"]

    def test_handler_error_does_not_stop_others(self) -> None:
        bus = EventBus()
        received: list[str] = []

        bus.on("test", lambda d: 1 / 0, priority=1)  # raises ZeroDivisionError
        bus.on("test", lambda d: received.append("ok"), priority=2)

        asyncio.get_event_loop().run_until_complete(bus.emit("test", None))
        assert received == ["ok"]

    def test_has_handlers(self) -> None:
        bus = EventBus()
        assert not bus.has_handlers("test")

        bus.on("test", lambda d: None)
        assert bus.has_handlers("test")
        assert not bus.has_handlers("other")

    def test_handler_count(self) -> None:
        bus = EventBus()
        assert bus.handler_count == 0

        bus.on("a", lambda d: None)
        bus.on("b", lambda d: None)
        assert bus.handler_count == 2

    def test_emit_no_handlers(self) -> None:
        bus = EventBus()
        results = asyncio.get_event_loop().run_until_complete(bus.emit("nope", None))
        assert results == []

    def test_emit_sync(self) -> None:
        bus = EventBus()
        received: list[str] = []

        bus.on("test", lambda d: received.append("sync"))
        bus.emit_sync("test", None)
        assert received == ["sync"]

    def test_emit_sync_skips_async(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def async_handler(d: object) -> None:
            received.append("async")

        bus.on("test", async_handler)
        bus.on("test", lambda d: received.append("sync"))
        bus.emit_sync("test", None)
        # async handler skipped, sync handler runs
        assert received == ["sync"]

    def test_multiple_unsubscribe_is_safe(self) -> None:
        bus = EventBus()
        unsub = bus.on("test", lambda d: None)
        unsub()
        unsub()  # second call should not raise
        assert bus.handler_count == 0


# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


class TestEventDataclasses:
    def test_agent_start_event(self) -> None:
        event = AgentStartEvent(
            user_input="hello", system_prompt="you are helpful", model="gpt-4"
        )
        assert event.user_input == "hello"
        assert event.model == "gpt-4"
        assert event.turn == 0

    def test_agent_end_event(self) -> None:
        event = AgentEndEvent(user_input="hello", total_turns=3, finish_reason="complete")
        assert event.total_turns == 3
        assert event.error is None

    def test_before_tool_call_event(self) -> None:
        event = BeforeToolCallEvent(
            tool_call_id="tc1", tool_name="execute", args={"command": "ls"}, turn=0
        )
        assert event.tool_name == "execute"

    def test_tool_call_event_result_defaults(self) -> None:
        result = ToolCallEventResult()
        assert result.block is False
        assert result.reason == ""
        assert result.modified_args is None

    def test_tool_result_event_result(self) -> None:
        result = ToolResultEventResult(modified_result="new output")
        assert result.modified_result == "new output"

    def test_context_transform_event_result(self) -> None:
        result = ContextTransformEventResult(messages=["msg1", "msg2"])
        assert result.messages == ["msg1", "msg2"]

    def test_input_event_result_defaults(self) -> None:
        result = InputEventResult()
        assert result.action == "continue"
        assert result.transformed_input is None
        assert result.response is None


# ---------------------------------------------------------------------------
# AgentRunner + EventBus integration tests
# ---------------------------------------------------------------------------


class TestAgentRunnerEvents:
    """Test that AgentRunner emits events at the right lifecycle points."""

    def _make_runner(self, events: EventBus | None = None) -> MagicMock:
        """Create a minimal AgentRunner with mocked LLM."""
        from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        agent_config = AgentConfig(
            model="test-model",
            system_prompt="test prompt",
            enable_tools=True,
            auto_execute=True,
        )
        runner = AgentRunner(engine, agent_config, events=events)
        return runner

    def test_runner_has_event_bus(self) -> None:
        runner = self._make_runner()
        assert isinstance(runner.events, EventBus)

    def test_runner_uses_provided_event_bus(self) -> None:
        bus = EventBus()
        runner = self._make_runner(events=bus)
        assert runner.events is bus

    def test_agent_start_end_events(self) -> None:
        """agent_start and agent_end should fire on a simple chat call."""
        from skillengine.agent import AgentMessage

        bus = EventBus()
        events_received: list[str] = []

        bus.on(AGENT_START, lambda e: events_received.append("start"))
        bus.on(AGENT_END, lambda e: events_received.append("end"))

        runner = self._make_runner(events=bus)

        # Mock _call_llm to return a simple response (no tool calls)
        response = AgentMessage(role="assistant", content="Hi there!")
        runner._call_llm = AsyncMock(return_value=response)

        asyncio.get_event_loop().run_until_complete(runner.chat("hello"))

        assert "start" in events_received
        assert "end" in events_received
        # start should come before end
        assert events_received.index("start") < events_received.index("end")

    def test_turn_events(self) -> None:
        """turn_start and turn_end should fire for each LLM round-trip."""
        from skillengine.agent import AgentMessage

        bus = EventBus()
        turns: list[tuple[str, int]] = []

        bus.on(TURN_START, lambda e: turns.append(("start", e.turn)))
        bus.on(TURN_END, lambda e: turns.append(("end", e.turn)))

        runner = self._make_runner(events=bus)

        response = AgentMessage(role="assistant", content="Done")
        runner._call_llm = AsyncMock(return_value=response)

        asyncio.get_event_loop().run_until_complete(runner.chat("test"))

        assert turns == [("start", 0), ("end", 0)]

    def test_before_tool_call_block(self) -> None:
        """before_tool_call handler can block a tool execution."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def block_dangerous(event: BeforeToolCallEvent) -> ToolCallEventResult:
            if "rm" in event.args.get("command", ""):
                return ToolCallEventResult(block=True, reason="No rm allowed")
            return ToolCallEventResult()

        bus.on(BEFORE_TOOL_CALL, block_dangerous)

        runner = self._make_runner(events=bus)

        # First call: LLM returns tool call for "rm -rf /"
        tool_response = AgentMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "execute",
                "arguments": '{"command": "rm -rf /"}',
            }],
        )
        # Second call: LLM sees blocked result and gives final answer
        final_response = AgentMessage(
            role="assistant",
            content="I cannot do that.",
        )
        runner._call_llm = AsyncMock(side_effect=[tool_response, final_response])

        result = asyncio.get_event_loop().run_until_complete(runner.chat("delete everything"))

        assert result.content == "I cannot do that."
        # The tool should NOT have been actually executed
        # Check conversation has the blocked message
        blocked_msgs = [
            m for m in runner._conversation if m.role == "tool" and "[Blocked]" in m.content
        ]
        assert len(blocked_msgs) == 1
        assert "No rm allowed" in blocked_msgs[0].content

    def test_before_tool_call_modify_args(self) -> None:
        """before_tool_call handler can modify tool arguments."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def add_safe_flag(event: BeforeToolCallEvent) -> ToolCallEventResult:
            if event.tool_name == "execute":
                new_args = dict(event.args)
                new_args["command"] = new_args.get("command", "") + " --safe"
                return ToolCallEventResult(modified_args=new_args)
            return ToolCallEventResult()

        bus.on(BEFORE_TOOL_CALL, add_safe_flag)

        runner = self._make_runner(events=bus)

        tool_response = AgentMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "execute",
                "arguments": '{"command": "deploy"}',
            }],
        )
        final_response = AgentMessage(role="assistant", content="Done")
        runner._call_llm = AsyncMock(side_effect=[tool_response, final_response])

        # Mock the engine execute to capture the actual command
        executed_commands: list[str] = []

        async def mock_execute(command: str, cwd: str | None = None, **kwargs) -> MagicMock:
            executed_commands.append(command)
            result = MagicMock()
            result.success = True
            result.output = "ok"
            return result

        runner.engine.execute = mock_execute

        asyncio.get_event_loop().run_until_complete(runner.chat("deploy it"))

        assert len(executed_commands) == 1
        assert executed_commands[0] == "deploy --safe"

    def test_after_tool_result_modify(self) -> None:
        """after_tool_result handler can modify the tool output."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def redact_secrets(event: AfterToolResultEvent) -> ToolResultEventResult:
            return ToolResultEventResult(
                modified_result=event.result.replace("SECRET123", "[REDACTED]")
            )

        bus.on(AFTER_TOOL_RESULT, redact_secrets)

        runner = self._make_runner(events=bus)

        tool_response = AgentMessage(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "execute",
                "arguments": '{"command": "echo SECRET123"}',
            }],
        )
        final_response = AgentMessage(role="assistant", content="Done")
        runner._call_llm = AsyncMock(side_effect=[tool_response, final_response])

        async def mock_execute(command: str, cwd: str | None = None, **kwargs) -> MagicMock:
            result = MagicMock()
            result.success = True
            result.output = "The key is SECRET123"
            return result

        runner.engine.execute = mock_execute

        asyncio.get_event_loop().run_until_complete(runner.chat("show secret"))

        tool_msgs = [m for m in runner._conversation if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "[REDACTED]" in tool_msgs[0].content
        assert "SECRET123" not in tool_msgs[0].content

    def test_input_event_transform(self) -> None:
        """input handler can transform user input."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def expand_alias(event: InputEvent) -> InputEventResult:
            if event.user_input == "hi":
                return InputEventResult(action="transform", transformed_input="Hello, how are you?")
            return InputEventResult()

        bus.on(INPUT, expand_alias)

        runner = self._make_runner(events=bus)
        response = AgentMessage(role="assistant", content="I'm fine!")
        runner._call_llm = AsyncMock(return_value=response)

        asyncio.get_event_loop().run_until_complete(runner.chat("hi"))

        # The conversation should contain the transformed input
        user_msgs = [m for m in runner._conversation if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hello, how are you?"

    def test_input_event_handled(self) -> None:
        """input handler can short-circuit and return a response directly."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def handle_help(event: InputEvent) -> InputEventResult:
            if event.user_input == "/myhelp":
                return InputEventResult(action="handled", response="Custom help text")
            return InputEventResult()

        bus.on(INPUT, handle_help)

        runner = self._make_runner(events=bus)
        # _call_llm should NOT be called
        runner._call_llm = AsyncMock(side_effect=AssertionError("LLM should not be called"))

        result = asyncio.get_event_loop().run_until_complete(runner.chat("/myhelp"))
        assert result.content == "Custom help text"

    def test_context_transform(self) -> None:
        """context_transform handler can modify messages before LLM call."""
        from skillengine.agent import AgentMessage

        bus = EventBus()

        def inject_reminder(event: ContextTransformEvent) -> ContextTransformEventResult:
            messages = list(event.messages)
            messages.append(AgentMessage(role="user", content="[System reminder: be concise]"))
            return ContextTransformEventResult(messages=messages)

        bus.on(CONTEXT_TRANSFORM, inject_reminder)

        runner = self._make_runner(events=bus)
        response = AgentMessage(role="assistant", content="Ok!")
        runner._call_llm = AsyncMock(return_value=response)

        asyncio.get_event_loop().run_until_complete(runner.chat("test"))

        # _call_llm should have been called with the injected message
        call_args = runner._call_llm.call_args[0][0]
        contents = [m.content for m in call_args]
        assert "[System reminder: be concise]" in contents

    def test_full_lifecycle_event_order(self) -> None:
        """Verify the full order of events in a simple chat with one tool call."""
        from skillengine.agent import AgentMessage

        bus = EventBus()
        event_log: list[str] = []

        bus.on(INPUT, lambda e: event_log.append("input"))
        bus.on(AGENT_START, lambda e: event_log.append("agent_start"))
        bus.on(TURN_START, lambda e: event_log.append(f"turn_start:{e.turn}"))
        bus.on(TURN_END, lambda e: event_log.append(f"turn_end:{e.turn}"))
        bus.on(BEFORE_TOOL_CALL, lambda e: event_log.append(f"before_tool:{e.tool_name}"))
        bus.on(AFTER_TOOL_RESULT, lambda e: event_log.append(f"after_tool:{e.tool_name}"))
        bus.on(AGENT_END, lambda e: event_log.append("agent_end"))

        runner = self._make_runner(events=bus)

        tool_response = AgentMessage(
            role="assistant",
            content="Let me check...",
            tool_calls=[{
                "id": "tc1",
                "name": "execute",
                "arguments": '{"command": "ls"}',
            }],
        )
        final_response = AgentMessage(role="assistant", content="Here are the files.")

        runner._call_llm = AsyncMock(side_effect=[tool_response, final_response])

        async def mock_execute(command: str, cwd: str | None = None, **kwargs) -> MagicMock:
            result = MagicMock()
            result.success = True
            result.output = "file1.txt"
            return result

        runner.engine.execute = mock_execute

        asyncio.get_event_loop().run_until_complete(runner.chat("list files"))

        assert event_log == [
            "input",
            "agent_start",
            "turn_start:0",
            "turn_end:0",
            "before_tool:execute",
            "after_tool:execute",
            "turn_start:1",
            "turn_end:1",
            "agent_end",
        ]

    def test_agent_end_on_error(self) -> None:
        """agent_end should fire even if an error occurs, with error info."""
        from skillengine.agent import AgentMessage

        bus = EventBus()
        end_events: list[AgentEndEvent] = []

        bus.on(AGENT_END, lambda e: end_events.append(e))

        runner = self._make_runner(events=bus)
        runner._call_llm = AsyncMock(side_effect=RuntimeError("LLM down"))

        with pytest.raises(RuntimeError, match="LLM down"):
            asyncio.get_event_loop().run_until_complete(runner.chat("test"))

        assert len(end_events) == 1
        assert end_events[0].finish_reason == "error"
        assert end_events[0].error == "LLM down"


# ---------------------------------------------------------------------------
# ExtensionManager + EventBus bridge tests
# ---------------------------------------------------------------------------


class TestExtensionManagerEventBridge:
    def test_extension_hook_registered_on_event_bus(self) -> None:
        """When ExtensionManager has an event_bus, hooks should register on both."""
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine
        from skillengine.extensions.manager import ExtensionManager

        bus = EventBus()
        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        manager = ExtensionManager(engine, event_bus=bus)

        handler = lambda **kwargs: "ok"
        manager._register_hook("test_event", handler, "test-ext", priority=5)

        # Should be in both the manager's hooks AND the event bus
        assert len(manager._hooks) == 1
        assert bus.has_handlers("test_event")
        assert bus.handler_count == 1

    def test_extension_hook_without_event_bus(self) -> None:
        """Without event_bus, hooks only register on the manager."""
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine
        from skillengine.extensions.manager import ExtensionManager

        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        manager = ExtensionManager(engine)  # no event_bus

        handler = lambda **kwargs: "ok"
        manager._register_hook("test_event", handler, "test-ext", priority=5)

        assert len(manager._hooks) == 1
