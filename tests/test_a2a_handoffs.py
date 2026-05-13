"""Tests for A2A-1: Handoffs compatibility shim (OpenAI Agents SDK / Anthropic A2A draft)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillengine import (
    Handoff,
    a2a_handoff,
    agent_handoff,
    callable_handoff,
    handoff,
)
from skillengine.a2a.handoffs import to_tool_definition
from skillengine.a2a.models import A2ATaskResponse, TaskStatus

# ---------------------------------------------------------------------------
# handoff() factory
# ---------------------------------------------------------------------------


def test_handoff_factory_defaults_to_transfer_to_slug() -> None:
    async def target(text: str, ctx: dict) -> str:
        return text

    h = handoff(target, name="Billing Bot")
    assert isinstance(h, Handoff)
    assert h.tool_name == "transfer_to_billing_bot"
    assert "Billing Bot" in h.description
    assert h.input_schema["type"] == "object"
    assert "input" in h.input_schema["properties"]


def test_handoff_factory_accepts_custom_tool_name_and_schema() -> None:
    async def target(text: str, ctx: dict) -> str:
        return "ok"

    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    h = handoff(
        target,
        name="Search",
        tool_name="ask_search",
        description="Search the corpus.",
        input_schema=schema,
    )
    assert h.tool_name == "ask_search"
    assert h.description == "Search the corpus."
    assert h.input_schema is schema


def test_handoff_name_slugifies_punctuation() -> None:
    async def target(text: str, ctx: dict) -> str:
        return text

    h = handoff(target, name="Customer  Support / Tier-2!!")
    assert h.tool_name == "transfer_to_customer_support_tier_2"


# ---------------------------------------------------------------------------
# callable_handoff()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callable_handoff_wraps_sync_function() -> None:
    h = callable_handoff(lambda s: f"got:{s}", name="sync-target")
    td = to_tool_definition(h)
    result = await td.handler("transfer_to_sync_target", {"input": "hi"}, None, None)
    assert result == "got:hi"


@pytest.mark.asyncio
async def test_callable_handoff_wraps_async_function() -> None:
    async def target(s: str) -> str:
        return f"async:{s}"

    h = callable_handoff(target, name="async-target")
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_async_target", {"input": "x"}, None, None)
    assert out == "async:x"


# ---------------------------------------------------------------------------
# to_tool_definition()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_tool_definition_fires_on_handoff_hook() -> None:
    calls: list[dict] = []

    def on_handoff(args: dict) -> None:
        calls.append(args)

    async def target(text: str, ctx: dict) -> str:
        return text.upper()

    h = handoff(target, name="upper", on_handoff=on_handoff)
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_upper", {"input": "hi"}, None, None)
    assert out == "HI"
    assert calls == [{"input": "hi"}]


@pytest.mark.asyncio
async def test_to_tool_definition_applies_input_filter() -> None:
    def keep_last(messages: list[dict]) -> list[dict]:
        return messages[-1:]

    captured: dict[str, Any] = {}

    async def target(text: str, ctx: dict) -> str:
        captured["messages"] = ctx.get("messages")
        return "done"

    h = handoff(target, name="trim", input_filter=keep_last)
    td = to_tool_definition(h)
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    await td.handler(
        "transfer_to_trim",
        {"input": "x"},
        {"messages": history},
        None,
    )
    assert captured["messages"] == [{"role": "assistant", "content": "b"}]


@pytest.mark.asyncio
async def test_to_tool_definition_accepts_alias_input_text() -> None:
    async def target(text: str, ctx: dict) -> str:
        return f"<{text}>"

    h = handoff(target, name="echo")
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_echo", {"input_text": "hello"}, None, None)
    assert out == "<hello>"


@pytest.mark.asyncio
async def test_to_tool_definition_returns_tool_definition_shape() -> None:
    async def target(text: str, ctx: dict) -> str:
        return text

    h = handoff(target, name="x", description="Transfer to x")
    td = to_tool_definition(h)
    assert td.name == "transfer_to_x"
    assert td.description == "Transfer to x"
    assert td.parameters == h.input_schema
    assert callable(td.handler)


# ---------------------------------------------------------------------------
# agent_handoff()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_handoff_calls_agent_chat() -> None:
    fake_agent = MagicMock()
    fake_agent.chat = AsyncMock(return_value=MagicMock(content="agent-reply"))

    h = agent_handoff(fake_agent, name="sub")
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_sub", {"input": "hi"}, None, None)
    assert out == "agent-reply"
    fake_agent.chat.assert_awaited_once_with("hi", reset=True)


@pytest.mark.asyncio
async def test_agent_handoff_passes_reset_flag() -> None:
    fake_agent = MagicMock()
    fake_agent.chat = AsyncMock(return_value=MagicMock(content="ok"))

    h = agent_handoff(fake_agent, name="keep", reset=False)
    td = to_tool_definition(h)
    await td.handler("transfer_to_keep", {"input": "q"}, None, None)
    fake_agent.chat.assert_awaited_once_with("q", reset=False)


# ---------------------------------------------------------------------------
# a2a_handoff()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2a_handoff_returns_remote_output() -> None:
    client = MagicMock()
    client.send_task = AsyncMock(
        return_value=A2ATaskResponse(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            output="remote-said-hi",
        )
    )
    h = a2a_handoff(client, endpoint="http://remote", skill_name="echo")
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_echo", {"input": "hi"}, None, None)
    assert out == "remote-said-hi"
    client.send_task.assert_awaited_once()
    kwargs = client.send_task.await_args.kwargs
    assert kwargs["endpoint"] == "http://remote"
    assert kwargs["skill_name"] == "echo"
    assert kwargs["input_text"] == "hi"


@pytest.mark.asyncio
async def test_a2a_handoff_surfaces_errors() -> None:
    client = MagicMock()
    client.send_task = AsyncMock(
        return_value=A2ATaskResponse(
            task_id="t1",
            status=TaskStatus.FAILED,
            output="",
            error="boom",
        )
    )
    h = a2a_handoff(client, endpoint="http://r", skill_name="x")
    td = to_tool_definition(h)
    out = await td.handler("transfer_to_x", {"input": "q"}, None, None)
    assert "[remote error]" in out and "boom" in out


# ---------------------------------------------------------------------------
# AgentRunner.add_handoffs / remove_handoffs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_runner_add_and_remove_handoffs() -> None:
    from skillengine.adapters.registry import AdapterRegistry
    from skillengine.agent import AgentConfig, AgentRunner
    from skillengine.engine import SkillsEngine
    from skillengine.events import EventBus

    engine = MagicMock(spec=SkillsEngine)
    engine.get_snapshot.return_value = MagicMock(skills=[], prompt="", get_skill=lambda n: None)
    config = AgentConfig(
        model="m",
        base_url="http://localhost",
        api_key="key",
        max_turns=1,
        enable_tools=True,
        auto_execute=False,
    )
    runner = AgentRunner(
        engine=engine,
        config=config,
        events=EventBus(),
        adapter_registry=AdapterRegistry(),
    )

    h1 = callable_handoff(lambda s: f"h1:{s}", name="alpha")
    h2 = callable_handoff(lambda s: f"h2:{s}", name="beta")
    names = runner.add_handoffs([h1, h2])
    assert names == ["transfer_to_alpha", "transfer_to_beta"]
    assert "transfer_to_alpha" in runner._dispatcher._tools
    assert "transfer_to_beta" in runner._dispatcher._tools

    # Invoke the registered handler through the dispatcher's stored tool.
    handler = runner._dispatcher._tools["transfer_to_alpha"].handler
    out = await handler("transfer_to_alpha", {"input": "x"}, None, None)
    assert out == "h1:x"

    removed = runner.remove_handoffs()
    assert removed == 2
    assert "transfer_to_alpha" not in runner._dispatcher._tools
    assert "transfer_to_beta" not in runner._dispatcher._tools
