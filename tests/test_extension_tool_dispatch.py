"""Tests for extension tool dispatch in AgentRunner."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.extensions.manager import ExtensionManager
from skillengine.extensions.models import ToolInfo


@pytest.fixture
def engine():
    return SkillsEngine()


@pytest.fixture
def runner(engine):
    config = AgentConfig(enable_tools=True)
    return AgentRunner(engine, config)


class TestGetToolsIncludesExtensionTools:
    def test_no_extensions_returns_hardcoded_only(self, runner):
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "execute" in names
        assert "execute_script" in names
        assert len(tools) == 4  # execute, execute_script, write, read

    def test_with_extension_tools(self, runner):
        runner.engine.extensions = ExtensionManager(runner.engine)
        runner.engine.extensions._register_tool(
            name="my_custom_tool",
            handler=lambda: "ok",
            description="A custom tool",
            parameters={"type": "object", "properties": {}},
            extension_name="test-ext",
        )

        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "execute" in names
        assert "execute_script" in names
        assert "my_custom_tool" in names
        assert len(tools) == 5  # 4 builtin + 1 extension

    def test_extension_tool_schema(self, runner):
        runner.engine.extensions = ExtensionManager(runner.engine)
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        runner.engine.extensions._register_tool(
            name="search_tool",
            handler=lambda query: f"found: {query}",
            description="Search for things",
            parameters=params,
            extension_name="test-ext",
        )

        tools = runner.get_tools()
        search_tool = next(t for t in tools if t["function"]["name"] == "search_tool")
        assert search_tool["function"]["description"] == "Search for things"
        assert search_tool["function"]["parameters"] == params

    def test_tools_disabled_returns_empty(self, runner):
        runner.config.enable_tools = False
        runner.engine.extensions = ExtensionManager(runner.engine)
        runner.engine.extensions._register_tool(
            name="my_tool",
            handler=lambda: "ok",
            description="Tool",
            parameters={},
            extension_name="test-ext",
        )
        assert runner.get_tools() == []


class TestExecuteToolDispatchesExtensionTools:
    @pytest.mark.asyncio
    async def test_sync_handler(self, runner):
        runner.engine.extensions = ExtensionManager(runner.engine)
        runner.engine.extensions._register_tool(
            name="greet",
            handler=lambda name="World": f"Hello, {name}!",
            description="Greet",
            parameters={},
            extension_name="test-ext",
        )

        result = await runner._execute_tool({
            "name": "greet",
            "arguments": json.dumps({"name": "Alice"}),
            "id": "call_1",
        })
        assert result == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_async_handler(self, runner):
        async def async_greet(name: str = "World") -> str:
            return f"Hello, {name}!"

        runner.engine.extensions = ExtensionManager(runner.engine)
        runner.engine.extensions._register_tool(
            name="async_greet",
            handler=async_greet,
            description="Async greet",
            parameters={},
            extension_name="test-ext",
        )

        result = await runner._execute_tool({
            "name": "async_greet",
            "arguments": json.dumps({"name": "Bob"}),
            "id": "call_2",
        })
        assert result == "Hello, Bob!"

    @pytest.mark.asyncio
    async def test_handler_returning_none(self, runner):
        runner.engine.extensions = ExtensionManager(runner.engine)
        runner.engine.extensions._register_tool(
            name="void_tool",
            handler=lambda: None,
            description="Returns nothing",
            parameters={},
            extension_name="test-ext",
        )

        result = await runner._execute_tool({
            "name": "void_tool",
            "arguments": "{}",
            "id": "call_3",
        })
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_unknown_tool(self, runner):
        result = await runner._execute_tool({
            "name": "nonexistent",
            "arguments": "{}",
            "id": "call_4",
        })
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_builtin_tools_still_work(self, runner):
        """Ensure execute and execute_script still dispatch correctly."""
        result = await runner._execute_tool({
            "name": "execute",
            "arguments": json.dumps({"command": "echo hello"}),
            "id": "call_5",
        })
        assert "hello" in result
