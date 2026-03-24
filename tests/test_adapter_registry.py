"""Tests for the dynamic adapter registry (Phase 4)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
from skillengine.adapters.registry import AdapterFactory, AdapterRegistry
from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.events import EventBus, StreamEvent
from skillengine.extensions.api import ExtensionAPI
from skillengine.extensions.manager import ExtensionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAdapter(LLMAdapter):
    """A minimal adapter for testing."""

    def __init__(self, engine: SkillsEngine, name: str = "fake") -> None:
        super().__init__(engine)
        self.name = name
        self.chat_calls: list[tuple] = []

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        **kwargs,
    ) -> AgentResponse:
        self.chat_calls.append((messages, system_prompt))
        return AgentResponse(
            content=f"response from {self.name}",
            finish_reason="stop",
        )

    async def chat_stream_events(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        **kwargs,
    ):
        yield StreamEvent(type="text_start")
        yield StreamEvent(type="text_delta", content=f"streamed from {self.name}")
        yield StreamEvent(type="text_end")
        yield StreamEvent(type="done", finish_reason="stop")


def _make_engine() -> MagicMock:
    engine = MagicMock(spec=SkillsEngine)
    engine.get_snapshot.return_value = MagicMock(
        skills=[], prompt="", get_skill=lambda n: None,
    )
    return engine


def _make_runner(
    adapter_registry: AdapterRegistry | None = None,
    **kwargs: Any,
) -> AgentRunner:
    engine = _make_engine()
    config = AgentConfig(
        model="test-model",
        base_url="http://localhost",
        api_key="test-key",
        max_turns=2,
        enable_tools=True,
        auto_execute=True,
    )
    return AgentRunner(
        engine=engine,
        config=config,
        events=EventBus(),
        adapter_registry=adapter_registry,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AdapterRegistry core
# ---------------------------------------------------------------------------


class TestAdapterRegistryBasic:
    def test_empty_registry(self) -> None:
        reg = AdapterRegistry()
        assert len(reg) == 0
        assert reg.list_adapters() == []
        assert reg.default_name is None
        assert reg.get_default() is None

    def test_register_and_get(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "a1")

        reg.register("a1", adapter)
        assert reg.has("a1")
        assert reg.get("a1") is adapter
        assert len(reg) == 1

    def test_register_sets_first_as_default(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("first", FakeAdapter(engine, "first"))
        reg.register("second", FakeAdapter(engine, "second"))

        assert reg.default_name == "first"

    def test_set_default(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a", FakeAdapter(engine, "a"))
        reg.register("b", FakeAdapter(engine, "b"))
        reg.set_default("b")

        assert reg.default_name == "b"

    def test_set_default_not_found_raises(self) -> None:
        reg = AdapterRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.set_default("nonexistent")

    def test_get_not_found_raises(self) -> None:
        reg = AdapterRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_register_empty_name_raises(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()
        with pytest.raises(ValueError, match="must not be empty"):
            reg.register("", FakeAdapter(engine))

    def test_override_adapter(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        a1 = FakeAdapter(engine, "v1")
        a2 = FakeAdapter(engine, "v2")

        reg.register("name", a1)
        reg.register("name", a2)

        assert reg.get("name") is a2
        assert len(reg) == 1


class TestAdapterRegistryFactory:
    def test_register_factory(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        def factory(eng: SkillsEngine) -> LLMAdapter:
            return FakeAdapter(eng, "from_factory")

        reg.register_factory("lazy", factory)

        assert reg.has("lazy")
        adapter = reg.get("lazy", engine=engine)
        assert isinstance(adapter, FakeAdapter)
        assert adapter.name == "from_factory"

    def test_factory_called_only_once(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()
        call_count = 0

        def factory(eng: SkillsEngine) -> LLMAdapter:
            nonlocal call_count
            call_count += 1
            return FakeAdapter(eng, "lazy")

        reg.register_factory("lazy", factory)

        a1 = reg.get("lazy", engine=engine)
        a2 = reg.get("lazy", engine=engine)

        assert call_count == 1
        assert a1 is a2

    def test_factory_without_engine_raises(self) -> None:
        reg = AdapterRegistry()

        reg.register_factory("lazy", lambda e: FakeAdapter(e))

        with pytest.raises(RuntimeError, match="requires a SkillsEngine"):
            reg.get("lazy")

    def test_factory_empty_name_raises(self) -> None:
        reg = AdapterRegistry()
        with pytest.raises(ValueError, match="must not be empty"):
            reg.register_factory("", lambda e: FakeAdapter(e))


class TestAdapterRegistryUnregister:
    def test_unregister(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a", FakeAdapter(engine))
        assert reg.unregister("a")
        assert not reg.has("a")
        assert len(reg) == 0

    def test_unregister_not_found(self) -> None:
        reg = AdapterRegistry()
        assert not reg.unregister("missing")

    def test_unregister_default_picks_next(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a", FakeAdapter(engine, "a"))
        reg.register("b", FakeAdapter(engine, "b"))
        assert reg.default_name == "a"

        reg.unregister("a")
        assert reg.default_name == "b"

    def test_unregister_by_source(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a1", FakeAdapter(engine), source="ext-a")
        reg.register("a2", FakeAdapter(engine), source="ext-a")
        reg.register("b1", FakeAdapter(engine), source="ext-b")

        removed = reg.unregister_by_source("ext-a")
        assert removed == 2
        assert not reg.has("a1")
        assert not reg.has("a2")
        assert reg.has("b1")


class TestAdapterRegistryInfo:
    def test_list_adapters(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("x", FakeAdapter(engine), source="src1")
        reg.register("y", FakeAdapter(engine), source="src2")

        assert sorted(reg.list_adapters()) == ["x", "y"]

    def test_list_by_source(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a", FakeAdapter(engine), source="ext")
        reg.register("b", FakeAdapter(engine), source="ext")
        reg.register("c", FakeAdapter(engine), source="other")

        assert sorted(reg.list_by_source("ext")) == ["a", "b"]

    def test_get_info(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("adapter1", FakeAdapter(engine), source="my-ext")

        info = reg.get_info("adapter1")
        assert info["name"] == "adapter1"
        assert info["source"] == "my-ext"
        assert info["has_instance"] is True
        assert info["has_factory"] is False
        assert info["is_default"] is True

    def test_get_info_not_found_raises(self) -> None:
        reg = AdapterRegistry()
        with pytest.raises(KeyError):
            reg.get_info("missing")

    def test_contains(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("x", FakeAdapter(engine))
        assert "x" in reg
        assert "y" not in reg

    def test_clear(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("a", FakeAdapter(engine))
        reg.register("b", FakeAdapter(engine))
        reg.clear()

        assert len(reg) == 0
        assert reg.default_name is None

    def test_repr(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()

        reg.register("openai", FakeAdapter(engine))
        s = repr(reg)
        assert "openai" in s
        assert "default=openai" in s


# ---------------------------------------------------------------------------
# AgentRunner integration
# ---------------------------------------------------------------------------


class TestAgentRunnerAdapterIntegration:
    def test_runner_has_adapter_registry(self) -> None:
        runner = _make_runner()
        assert isinstance(runner.adapter_registry, AdapterRegistry)

    def test_runner_active_adapter_none_when_empty(self) -> None:
        runner = _make_runner()
        assert runner.active_adapter is None

    def test_set_adapter(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "my-adapter")
        reg.register("my-adapter", adapter)

        runner = _make_runner(adapter_registry=reg)
        runner.set_adapter("my-adapter")

        assert runner.active_adapter is adapter

    def test_set_adapter_not_found_raises(self) -> None:
        runner = _make_runner()
        with pytest.raises(KeyError, match="not found"):
            runner.set_adapter("nonexistent")

    def test_active_adapter_uses_default(self) -> None:
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "default-one")
        reg.register("default-one", adapter)

        runner = _make_runner(adapter_registry=reg)
        # No set_adapter call — uses default
        assert runner.active_adapter is adapter

    @pytest.mark.asyncio
    async def test_call_llm_delegates_to_adapter(self) -> None:
        """When adapter is active, _call_llm should use adapter.chat()."""
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "test")
        reg.register("test", adapter)

        runner = _make_runner(adapter_registry=reg)

        messages = [AgentMessage(role="user", content="hello")]
        result = await runner._call_llm(messages)

        assert result.content == "response from test"
        assert len(adapter.chat_calls) == 1

    @pytest.mark.asyncio
    async def test_call_llm_stream_delegates_to_adapter(self) -> None:
        """When adapter is active, _call_llm_stream should use adapter."""
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "stream-test")
        reg.register("stream-test", adapter)

        runner = _make_runner(adapter_registry=reg)

        messages = [AgentMessage(role="user", content="hello")]
        events: list[StreamEvent] = []
        async for event in runner._call_llm_stream(messages):
            events.append(event)

        types = [e.type for e in events]
        assert "text_start" in types
        assert "text_delta" in types
        assert "text_end" in types
        # Check content
        text_deltas = [e for e in events if e.type == "text_delta"]
        assert text_deltas[0].content == "streamed from stream-test"

    @pytest.mark.asyncio
    async def test_chat_with_adapter(self) -> None:
        """Full chat() flow with adapter."""
        reg = AdapterRegistry()
        engine = _make_engine()
        adapter = FakeAdapter(engine, "full-test")
        reg.register("full-test", adapter)

        runner = _make_runner(adapter_registry=reg)

        result = await runner.chat("test message")
        assert result.content == "response from full-test"

    @pytest.mark.asyncio
    async def test_switch_adapter_mid_session(self) -> None:
        """Can switch adapters between calls."""
        reg = AdapterRegistry()
        engine = _make_engine()

        adapter_a = FakeAdapter(engine, "adapter-a")
        adapter_b = FakeAdapter(engine, "adapter-b")
        reg.register("a", adapter_a)
        reg.register("b", adapter_b)

        runner = _make_runner(adapter_registry=reg)

        runner.set_adapter("a")
        result1 = await runner.chat("msg1", reset=True)
        assert result1.content == "response from adapter-a"

        runner.set_adapter("b")
        result2 = await runner.chat("msg2", reset=True)
        assert result2.content == "response from adapter-b"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_convert_user_message(self) -> None:
        runner = _make_runner()
        msgs = [AgentMessage(role="user", content="hello")]
        converted = runner._convert_to_adapter_messages(msgs)
        assert len(converted) == 1
        assert converted[0].role == "user"
        assert converted[0].content == "hello"

    def test_convert_tool_message(self) -> None:
        runner = _make_runner()
        msgs = [AgentMessage(
            role="tool", content="result",
            tool_call_id="tc1", name="execute",
        )]
        converted = runner._convert_to_adapter_messages(msgs)
        assert converted[0].metadata["tool_call_id"] == "tc1"
        assert converted[0].metadata["name"] == "execute"

    def test_convert_assistant_with_tool_calls(self) -> None:
        runner = _make_runner()
        tc = [{"id": "tc1", "name": "execute", "arguments": '{"command":"ls"}'}]
        msgs = [AgentMessage(role="assistant", content="", tool_calls=tc)]
        converted = runner._convert_to_adapter_messages(msgs)
        assert converted[0].metadata["tool_calls"] == tc

    def test_adapter_response_to_agent_message(self) -> None:
        runner = _make_runner()
        from skillengine.model_registry import TokenUsage

        response = AgentResponse(
            content="result",
            tool_calls=[{"id": "tc1", "name": "exec", "arguments": "{}"}],
            finish_reason="stop",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        msg = runner._adapter_response_to_agent_message(response)
        assert msg.role == "assistant"
        assert msg.content == "result"
        assert msg.tool_calls == response.tool_calls
        assert msg.token_usage.input_tokens == 10


# ---------------------------------------------------------------------------
# ExtensionAPI.register_adapter
# ---------------------------------------------------------------------------


class TestExtensionAdapterRegistration:
    def test_extension_registers_adapter_instance(self) -> None:
        engine = _make_engine()
        reg = AdapterRegistry()
        manager = ExtensionManager(engine, adapter_registry=reg)
        api = ExtensionAPI(manager, extension_name="my-ext")

        adapter = FakeAdapter(engine, "ext-adapter")
        api.register_adapter("ext-adapter", adapter=adapter)

        assert reg.has("ext-adapter")
        assert reg.get("ext-adapter") is adapter

    def test_extension_registers_adapter_factory(self) -> None:
        engine = _make_engine()
        reg = AdapterRegistry()
        manager = ExtensionManager(engine, adapter_registry=reg)
        api = ExtensionAPI(manager, extension_name="my-ext")

        api.register_adapter(
            "lazy-adapter",
            factory=lambda eng: FakeAdapter(eng, "lazy"),
        )

        assert reg.has("lazy-adapter")
        adapter = reg.get("lazy-adapter", engine=engine)
        assert isinstance(adapter, FakeAdapter)
        assert adapter.name == "lazy"

    def test_extension_adapter_source_tracking(self) -> None:
        engine = _make_engine()
        reg = AdapterRegistry()
        manager = ExtensionManager(engine, adapter_registry=reg)
        api = ExtensionAPI(manager, extension_name="test-ext")

        api.register_adapter("a1", adapter=FakeAdapter(engine))

        info = reg.get_info("a1")
        assert info["source"] == "test-ext"

    def test_extension_unregister_by_source(self) -> None:
        engine = _make_engine()
        reg = AdapterRegistry()
        manager = ExtensionManager(engine, adapter_registry=reg)

        api1 = ExtensionAPI(manager, extension_name="ext-a")
        api2 = ExtensionAPI(manager, extension_name="ext-b")

        api1.register_adapter("a1", adapter=FakeAdapter(engine))
        api2.register_adapter("b1", adapter=FakeAdapter(engine))

        removed = reg.unregister_by_source("ext-a")
        assert removed == 1
        assert not reg.has("a1")
        assert reg.has("b1")

    def test_no_registry_logs_warning(self) -> None:
        """Manager without adapter_registry should not crash."""
        engine = _make_engine()
        manager = ExtensionManager(engine)  # No adapter_registry
        api = ExtensionAPI(manager, extension_name="test")

        # Should not raise — just logs warning
        api.register_adapter("x", adapter=FakeAdapter(engine))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestPhase4Exports:
    def test_adapter_registry_exported(self) -> None:
        from skillengine import AdapterRegistry
        assert AdapterRegistry is not None

    def test_adapter_factory_exported(self) -> None:
        from skillengine import AdapterFactory
        assert AdapterFactory is not None
