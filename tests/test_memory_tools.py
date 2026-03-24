"""Tests for memory tool handlers."""

from unittest.mock import AsyncMock

import pytest

from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig
from skillengine.memory.tools import (
    MemoryState,
    build_memory_tools,
    make_add_knowledge_handler,
    make_explore_handler,
    make_recall_handler,
    make_save_handler,
)


@pytest.fixture
def client():
    c = OpenVikingClient(MemoryConfig())
    c.available = True
    c._client = AsyncMock()
    return c


@pytest.fixture
def state(client):
    s = MemoryState(client)
    s.session_id = "sess-test"
    return s


class TestRecallMemory:
    @pytest.mark.asyncio
    async def test_uses_search_with_session(self, state, client):
        client.search = AsyncMock(return_value=[{"content": "user prefers dark mode", "score": 0.9}])
        handler = make_recall_handler(state)

        result = await handler(query="dark mode preference")
        assert "dark mode" in result
        client.search.assert_called_once_with(
            query="dark mode preference",
            target_uri="viking://user/memories/",
            session_id="sess-test",
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_uses_find_without_session(self, state, client):
        state.session_id = None
        client.find = AsyncMock(return_value=[{"content": "found"}])
        handler = make_recall_handler(state)

        result = await handler(query="test")
        assert "found" in result
        client.find.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_scope(self, state, client):
        client.search = AsyncMock(return_value=[])
        handler = make_recall_handler(state)

        await handler(query="test", scope="agent")
        call_args = client.search.call_args
        assert call_args[1]["target_uri"] == "viking://agent/memories/"

    @pytest.mark.asyncio
    async def test_custom_limit(self, state, client):
        client.search = AsyncMock(return_value=[])
        handler = make_recall_handler(state)

        await handler(query="test", limit=10)
        call_args = client.search.call_args
        assert call_args[1]["limit"] == 10

    @pytest.mark.asyncio
    async def test_unavailable(self, state, client):
        client.available = False
        handler = make_recall_handler(state)

        result = await handler(query="test")
        assert result == "[Memory unavailable]"

    @pytest.mark.asyncio
    async def test_search_returns_none(self, state, client):
        client.search = AsyncMock(return_value=None)
        handler = make_recall_handler(state)

        result = await handler(query="test")
        assert result == "[Memory unavailable]"

    @pytest.mark.asyncio
    async def test_no_results(self, state, client):
        client.search = AsyncMock(return_value=[])
        handler = make_recall_handler(state)

        result = await handler(query="test")
        assert result == "[No memories found]"


class TestSaveMemory:
    @pytest.mark.asyncio
    async def test_saves_and_commits(self, state, client):
        client.add_message = AsyncMock(return_value=True)
        client.commit_session = AsyncMock(return_value=True)
        handler = make_save_handler(state)

        result = await handler(content="User likes Python", category="preferences")
        assert "Memory saved" in result
        assert "preferences" in result
        client.add_message.assert_called_once_with(
            "sess-test", "assistant", "[memory:preferences] User likes Python"
        )
        client.commit_session.assert_called_once_with("sess-test")

    @pytest.mark.asyncio
    async def test_save_without_commit(self, state, client):
        client.add_message = AsyncMock(return_value=True)
        client.commit_session = AsyncMock(return_value=False)
        handler = make_save_handler(state)

        result = await handler(content="data")
        assert "commit pending" in result

    @pytest.mark.asyncio
    async def test_no_session(self, state, client):
        state.session_id = None
        handler = make_save_handler(state)

        result = await handler(content="data")
        assert result == "[No active memory session]"

    @pytest.mark.asyncio
    async def test_unavailable(self, state, client):
        client.available = False
        handler = make_save_handler(state)

        result = await handler(content="data")
        assert result == "[Memory unavailable]"

    @pytest.mark.asyncio
    async def test_add_message_fails(self, state, client):
        client.add_message = AsyncMock(return_value=False)
        handler = make_save_handler(state)

        result = await handler(content="data")
        assert result == "[Failed to save memory]"


class TestExploreMemory:
    @pytest.mark.asyncio
    async def test_list_entries(self, state, client):
        client.ls = AsyncMock(return_value=[
            {"name": "preferences", "type": "directory"},
            {"name": "entities", "type": "directory"},
        ])
        handler = make_explore_handler(state)

        result = await handler()
        assert "preferences" in result
        assert "entities" in result
        assert "[dir]" in result

    @pytest.mark.asyncio
    async def test_empty(self, state, client):
        client.ls = AsyncMock(return_value=[])
        handler = make_explore_handler(state)

        result = await handler()
        assert result == "[Empty]"

    @pytest.mark.asyncio
    async def test_unavailable(self, state, client):
        client.available = False
        handler = make_explore_handler(state)

        result = await handler()
        assert result == "[Memory unavailable]"

    @pytest.mark.asyncio
    async def test_ls_fails(self, state, client):
        client.ls = AsyncMock(return_value=None)
        handler = make_explore_handler(state)

        result = await handler()
        assert result == "[Memory unavailable]"

    @pytest.mark.asyncio
    async def test_custom_uri(self, state, client):
        client.ls = AsyncMock(return_value=[])
        handler = make_explore_handler(state)

        await handler(uri="viking://agent/memories/")
        client.ls.assert_called_once_with(uri="viking://agent/memories/", recursive=False)


class TestAddKnowledge:
    @pytest.mark.asyncio
    async def test_success(self, state, client):
        client.add_resource = AsyncMock(return_value="viking://knowledge/file.py")
        handler = make_add_knowledge_handler(state)

        result = await handler(path="/code/file.py", reason="core logic")
        assert "Indexed" in result
        assert "viking://knowledge/file.py" in result

    @pytest.mark.asyncio
    async def test_failure(self, state, client):
        client.add_resource = AsyncMock(return_value=None)
        handler = make_add_knowledge_handler(state)

        result = await handler(path="/code/file.py")
        assert "Failed" in result

    @pytest.mark.asyncio
    async def test_unavailable(self, state, client):
        client.available = False
        handler = make_add_knowledge_handler(state)

        result = await handler(path="/code/file.py")
        assert result == "[Memory unavailable]"


class TestBuildMemoryTools:
    def test_returns_four_tools(self, state):
        tools = build_memory_tools(state)
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "recall_memory" in names
        assert "save_memory" in names
        assert "explore_memory" in names
        assert "add_knowledge" in names

    def test_all_have_handlers(self, state):
        tools = build_memory_tools(state)
        for tool in tools:
            assert tool.handler is not None
            assert callable(tool.handler)

    def test_all_have_parameters(self, state):
        tools = build_memory_tools(state)
        for tool in tools:
            assert isinstance(tool.parameters, dict)
            assert tool.parameters.get("type") == "object"
