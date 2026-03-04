"""Tests for setup_memory() integration wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillkit.agent import AgentConfig, AgentRunner
from skillkit.engine import SkillsEngine
from skillkit.events import AGENT_END, AGENT_START, CONTEXT_TRANSFORM, EventBus
from skillkit.memory.config import MemoryConfig
from skillkit.memory.extension import setup_memory


@pytest.fixture
def engine():
    return SkillsEngine()


@pytest.fixture
def runner(engine):
    config = AgentConfig(enable_tools=True)
    return AgentRunner(engine, config)


class TestSetupMemory:
    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self, runner):
        """setup_memory returns None when OV is not reachable."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            result = await setup_memory(runner, MemoryConfig())
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_client_when_available(self, runner):
        """setup_memory returns the client when OV is reachable."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            result = await setup_memory(runner, MemoryConfig())
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_registers_four_tools(self, runner):
        """setup_memory registers 4 memory tools."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            await setup_memory(runner, MemoryConfig())

            assert runner.engine.extensions is not None
            tools = runner.engine.extensions.get_tools()
            tool_names = [t.name for t in tools]
            assert "recall_memory" in tool_names
            assert "save_memory" in tool_names
            assert "explore_memory" in tool_names
            assert "add_knowledge" in tool_names

    @pytest.mark.asyncio
    async def test_tools_appear_in_get_tools(self, runner):
        """Extension tools should appear in AgentRunner.get_tools()."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            await setup_memory(runner, MemoryConfig())

            tools = runner.get_tools()
            tool_names = [t["function"]["name"] for t in tools]
            assert "recall_memory" in tool_names
            assert "save_memory" in tool_names
            assert "explore_memory" in tool_names
            assert "add_knowledge" in tool_names
            # Plus the 5 hardcoded tools (execute, execute_script, write, read, apply_patch)
            assert "execute" in tool_names
            assert "execute_script" in tool_names
            assert "apply_patch" in tool_names
            assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_registers_event_hooks(self, runner):
        """setup_memory registers 3 event hooks on the EventBus."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            await setup_memory(runner, MemoryConfig())

            assert runner.events.has_handlers(AGENT_START)
            assert runner.events.has_handlers(CONTEXT_TRANSFORM)
            assert runner.events.has_handlers(AGENT_END)

    @pytest.mark.asyncio
    async def test_creates_extension_manager_if_needed(self, runner):
        """If engine has no ExtensionManager, setup_memory creates one."""
        assert runner.engine.extensions is None

        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            await setup_memory(runner, MemoryConfig())

            assert runner.engine.extensions is not None

    @pytest.mark.asyncio
    async def test_reuses_existing_extension_manager(self, runner):
        """If engine already has an ExtensionManager, setup_memory reuses it."""
        runner.engine.init_extensions()
        original_manager = runner.engine.extensions

        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            await setup_memory(runner, MemoryConfig())

            assert runner.engine.extensions is original_manager

    @pytest.mark.asyncio
    async def test_default_config(self, runner):
        """setup_memory uses default config when None is passed."""
        with patch(
            "skillkit.memory.extension.OpenVikingClient"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock(return_value=True)
            mock_instance.available = True
            MockClient.return_value = mock_instance

            result = await setup_memory(runner)
            assert result is mock_instance
