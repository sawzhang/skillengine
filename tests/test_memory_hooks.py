"""Tests for memory lifecycle hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from skillengine.events import AgentEndEvent, AgentStartEvent, ContextTransformEvent
from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig
from skillengine.memory.hooks import MemoryHooks
from skillengine.memory.tools import MemoryState


@pytest.fixture
def client():
    c = OpenVikingClient(MemoryConfig())
    c.available = True
    c._client = AsyncMock()
    return c


@pytest.fixture
def state(client):
    return MemoryState(client)


@pytest.fixture
def config():
    return MemoryConfig()


def make_message(role: str, content: str):
    """Create a minimal message-like object."""
    return MagicMock(role=role, content=content)


class TestOnAgentStart:
    @pytest.mark.asyncio
    async def test_creates_session(self, client, config, state):
        client.create_session = AsyncMock(return_value="sess-new")
        conversation = []
        hooks = MemoryHooks(client, config, state, lambda: conversation)
        event = AgentStartEvent(user_input="hi", system_prompt="", model="test")

        await hooks.on_agent_start(event)

        assert state.session_id == "sess-new"
        assert hooks._synced_message_count == 0
        client.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_auto_session_false(self, client, state):
        cfg = MemoryConfig(auto_session=False)
        hooks = MemoryHooks(client, cfg, state, lambda: [])
        event = AgentStartEvent(user_input="hi", system_prompt="", model="test")

        await hooks.on_agent_start(event)
        assert state.session_id is None

    @pytest.mark.asyncio
    async def test_skips_when_unavailable(self, client, config, state):
        client.available = False
        hooks = MemoryHooks(client, config, state, lambda: [])
        event = AgentStartEvent(user_input="hi", system_prompt="", model="test")

        await hooks.on_agent_start(event)
        assert state.session_id is None

    @pytest.mark.asyncio
    async def test_handles_session_creation_failure(self, client, config, state):
        client.create_session = AsyncMock(return_value=None)
        hooks = MemoryHooks(client, config, state, lambda: [])
        event = AgentStartEvent(user_input="hi", system_prompt="", model="test")

        await hooks.on_agent_start(event)
        assert state.session_id is None


class TestOnContextTransform:
    @pytest.mark.asyncio
    async def test_syncs_new_messages(self, client, config, state):
        state.session_id = "sess-1"
        client.add_message = AsyncMock(return_value=True)

        messages = [
            make_message("user", "hello"),
            make_message("assistant", "hi there"),
        ]
        hooks = MemoryHooks(client, config, state, lambda: messages)
        event = ContextTransformEvent(messages=messages, turn=0)

        await hooks.on_context_transform(event)

        assert client.add_message.call_count == 2
        assert hooks._synced_message_count == 2

    @pytest.mark.asyncio
    async def test_only_syncs_new_messages(self, client, config, state):
        state.session_id = "sess-1"
        client.add_message = AsyncMock(return_value=True)

        messages = [
            make_message("user", "first"),
            make_message("assistant", "response"),
        ]
        hooks = MemoryHooks(client, config, state, lambda: messages)
        hooks._synced_message_count = 1  # Already synced first message

        event = ContextTransformEvent(messages=messages, turn=0)
        await hooks.on_context_transform(event)

        # Only the second message should be synced
        client.add_message.assert_called_once()
        assert hooks._synced_message_count == 2

    @pytest.mark.asyncio
    async def test_skips_tool_messages(self, client, config, state):
        state.session_id = "sess-1"
        client.add_message = AsyncMock(return_value=True)

        messages = [
            make_message("user", "do something"),
            make_message("tool", "tool output"),
            make_message("assistant", "done"),
        ]
        hooks = MemoryHooks(client, config, state, lambda: messages)
        event = ContextTransformEvent(messages=messages, turn=0)

        await hooks.on_context_transform(event)

        # Only user and assistant messages synced (not tool)
        assert client.add_message.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_when_no_session(self, client, config, state):
        state.session_id = None
        client.add_message = AsyncMock()
        hooks = MemoryHooks(client, config, state, lambda: [make_message("user", "hi")])
        event = ContextTransformEvent(messages=[], turn=0)

        await hooks.on_context_transform(event)
        client.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_auto_sync_false(self, client, state):
        state.session_id = "sess-1"
        cfg = MemoryConfig(auto_sync=False)
        client.add_message = AsyncMock()
        hooks = MemoryHooks(client, cfg, state, lambda: [make_message("user", "hi")])
        event = ContextTransformEvent(messages=[], turn=0)

        await hooks.on_context_transform(event)
        client.add_message.assert_not_called()


class TestOnAgentEnd:
    @pytest.mark.asyncio
    async def test_syncs_and_commits(self, client, config, state):
        state.session_id = "sess-1"
        client.add_message = AsyncMock(return_value=True)
        client.commit_session = AsyncMock(return_value=True)

        messages = [
            make_message("user", "bye"),
            make_message("assistant", "goodbye"),
        ]
        hooks = MemoryHooks(client, config, state, lambda: messages)
        event = AgentEndEvent(
            user_input="bye", total_turns=1, finish_reason="complete"
        )

        await hooks.on_agent_end(event)

        assert client.add_message.call_count == 2
        client.commit_session.assert_called_once_with("sess-1")

    @pytest.mark.asyncio
    async def test_no_commit_when_disabled(self, client, state):
        state.session_id = "sess-1"
        cfg = MemoryConfig(auto_commit=False)
        client.add_message = AsyncMock(return_value=True)
        client.commit_session = AsyncMock()

        hooks = MemoryHooks(client, cfg, state, lambda: [])
        event = AgentEndEvent(
            user_input="bye", total_turns=1, finish_reason="complete"
        )

        await hooks.on_agent_end(event)
        client.commit_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_session_skips_all(self, client, config, state):
        state.session_id = None
        client.add_message = AsyncMock()
        client.commit_session = AsyncMock()

        hooks = MemoryHooks(client, config, state, lambda: [])
        event = AgentEndEvent(
            user_input="bye", total_turns=1, finish_reason="complete"
        )

        await hooks.on_agent_end(event)
        client.add_message.assert_not_called()
        client.commit_session.assert_not_called()
