"""Tests for the unified ToolDispatcher (Sprint 1 of Managed Agents refactoring)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillengine.tools.dispatcher import OutputCallback, ToolContext, ToolDispatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides: Any) -> ToolContext:
    """Create a minimal ToolContext for testing."""
    defaults = dict(
        engine=MagicMock(),
        config=MagicMock(session_id="test-session", model="test-model"),
        abort_event=asyncio.Event(),
        event_bus=MagicMock(),
        turn=0,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


async def _echo_handler(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    on_output: OutputCallback | None = None,
) -> str:
    """Simple test handler that echoes back the args."""
    return f"echo:{name}:{args}"


async def _pattern_handler(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    on_output: OutputCallback | None = None,
) -> str:
    """Pattern handler for skill:action names."""
    return f"pattern:{name}:{args}"


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestToolDispatcherRegistration:
    def test_register_and_has(self):
        d = ToolDispatcher()
        assert not d.has("my_tool")

        d.register("my_tool", _echo_handler, {"type": "function", "function": {"name": "my_tool"}})
        assert d.has("my_tool")

    def test_unregister(self):
        d = ToolDispatcher()
        d.register("my_tool", _echo_handler, {"type": "function", "function": {"name": "my_tool"}})
        assert d.unregister("my_tool") is True
        assert not d.has("my_tool")
        assert d.unregister("my_tool") is False

    def test_clear(self):
        d = ToolDispatcher()
        d.register("a", _echo_handler, {"type": "function", "function": {"name": "a"}})
        d.register("b", _echo_handler, {"type": "function", "function": {"name": "b"}})
        d.clear()
        assert not d.has("a")
        assert not d.has("b")

    def test_get_definitions(self):
        d = ToolDispatcher()
        defn = {"type": "function", "function": {"name": "my_tool", "description": "test"}}
        d.register("my_tool", _echo_handler, defn)
        defs = d.get_definitions()
        assert len(defs) == 1
        assert defs[0] == defn

    def test_get_definition_by_name(self):
        d = ToolDispatcher()
        defn = {"type": "function", "function": {"name": "my_tool"}}
        d.register("my_tool", _echo_handler, defn)
        assert d.get_definition("my_tool") == defn
        assert d.get_definition("nonexistent") is None


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestToolDispatcherDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_exact_match(self):
        d = ToolDispatcher()
        d.register("my_tool", _echo_handler, {})
        ctx = _make_ctx()
        result = await d.dispatch("my_tool", {"key": "val"}, ctx)
        assert "echo:my_tool:" in result
        assert "'key': 'val'" in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        d = ToolDispatcher()
        ctx = _make_ctx()
        result = await d.dispatch("nonexistent", {}, ctx)
        assert "Error: Unknown tool 'nonexistent'" == result

    @pytest.mark.asyncio
    async def test_dispatch_pattern_handler(self):
        d = ToolDispatcher()
        d.register_pattern_handler(_pattern_handler)
        ctx = _make_ctx()
        result = await d.dispatch("skill:action", {"x": 1}, ctx)
        assert "pattern:skill:action:" in result

    @pytest.mark.asyncio
    async def test_pattern_handler_not_called_for_exact_match(self):
        """Exact match takes precedence over pattern handler."""
        d = ToolDispatcher()
        d.register("a:b", _echo_handler, {})
        d.register_pattern_handler(_pattern_handler)
        ctx = _make_ctx()
        result = await d.dispatch("a:b", {}, ctx)
        assert result.startswith("echo:")

    @pytest.mark.asyncio
    async def test_colon_name_without_pattern_handler(self):
        """Colon name with no pattern handler returns unknown."""
        d = ToolDispatcher()
        ctx = _make_ctx()
        result = await d.dispatch("a:b", {}, ctx)
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_on_output_forwarded(self):
        """on_output callback is forwarded to handler."""
        received_output = []

        async def capturing_handler(name, args, ctx, on_output=None):
            if on_output:
                on_output("line1")
            return "done"

        d = ToolDispatcher()
        d.register("cap", capturing_handler, {})
        ctx = _make_ctx()
        result = await d.dispatch("cap", {}, ctx, on_output=received_output.append)
        assert result == "done"
        assert received_output == ["line1"]


# ---------------------------------------------------------------------------
# Built-in registration tests
# ---------------------------------------------------------------------------


class TestRegisterBuiltins:
    def test_register_builtins_adds_six_tools(self):
        d = ToolDispatcher()
        engine = MagicMock()
        d.register_builtins(engine)
        expected = {"execute", "execute_script", "write", "read", "edit", "apply_patch"}
        registered = {defn["function"]["name"] for defn in d.get_definitions()}
        assert registered == expected

    @pytest.mark.asyncio
    async def test_builtin_write_creates_file(self, tmp_path):
        d = ToolDispatcher()
        d.register_builtins(MagicMock())
        ctx = _make_ctx()
        target = str(tmp_path / "test.txt")
        result = await d.dispatch("write", {"path": target, "content": "hello"}, ctx)
        assert "Written 5 bytes" in result
        assert (tmp_path / "test.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_builtin_read_returns_content(self, tmp_path):
        d = ToolDispatcher()
        d.register_builtins(MagicMock())
        ctx = _make_ctx()
        target = tmp_path / "test.txt"
        target.write_text("world")
        result = await d.dispatch("read", {"path": str(target)}, ctx)
        assert result == "world"

    @pytest.mark.asyncio
    async def test_builtin_read_missing_file(self):
        d = ToolDispatcher()
        d.register_builtins(MagicMock())
        ctx = _make_ctx()
        result = await d.dispatch("read", {"path": "/nonexistent/file.txt"}, ctx)
        assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_builtin_execute_delegates_to_engine(self):
        d = ToolDispatcher()
        engine = MagicMock()
        engine.env_context = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        exec_result = MagicMock(success=True, output="hello world", exit_code=0)
        engine.execute = AsyncMock(return_value=exec_result)
        d.register_builtins(engine)
        ctx = _make_ctx(engine=engine)
        result = await d.dispatch("execute", {"command": "echo hello"}, ctx)
        assert result == "hello world"
        engine.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Skill tool registration tests
# ---------------------------------------------------------------------------


class TestRegisterSkillTools:
    def test_register_skill_tools_adds_skill_tool(self):
        d = ToolDispatcher()
        skill = MagicMock()
        skill.name = "my-skill"
        skill.metadata.invocation.disable_model_invocation = False
        d.register_skill_tools([skill], _echo_handler)
        assert d.has("skill")
        defn = d.get_definition("skill")
        assert "my-skill" in defn["function"]["parameters"]["properties"]["name"]["description"]

    def test_hidden_skills_excluded(self):
        d = ToolDispatcher()
        skill = MagicMock()
        skill.name = "hidden"
        skill.metadata.invocation.disable_model_invocation = True
        d.register_skill_tools([skill], _echo_handler)
        assert not d.has("skill")

    def test_register_action_tools(self):
        d = ToolDispatcher()
        d.register_pattern_handler(_pattern_handler)

        skill = MagicMock()
        skill.name = "pdf"
        skill.has_actions = True
        action = MagicMock()
        action.name = "convert"
        action.description = "Convert PDF"
        action.params = []
        skill.actions = {"convert": action}
        d.register_action_tools([skill])
        assert d.has("pdf:convert")


# ---------------------------------------------------------------------------
# Extension tool registration tests
# ---------------------------------------------------------------------------


class TestRegisterExtensionTools:
    def test_register_extension_tools(self):
        d = ToolDispatcher()
        ext_mgr = MagicMock()
        tool_info = MagicMock()
        tool_info.name = "ext_tool"
        tool_info.description = "An extension tool"
        tool_info.parameters = {"type": "object", "properties": {}}
        tool_info.handler = MagicMock(return_value="ext_result")
        ext_mgr.get_tools.return_value = [tool_info]
        d.register_extension_tools(ext_mgr)
        assert d.has("ext_tool")

    @pytest.mark.asyncio
    async def test_extension_tool_dispatch(self):
        d = ToolDispatcher()
        ext_mgr = MagicMock()
        tool_info = MagicMock()
        tool_info.name = "ext_tool"
        tool_info.description = "test"
        tool_info.parameters = {}
        tool_info.handler = MagicMock(return_value="ext_result")
        ext_mgr.get_tools.return_value = [tool_info]
        d.register_extension_tools(ext_mgr)
        ctx = _make_ctx()
        result = await d.dispatch("ext_tool", {}, ctx)
        assert result == "ext_result"

    @pytest.mark.asyncio
    async def test_extension_tool_async_handler(self):
        d = ToolDispatcher()
        ext_mgr = MagicMock()
        tool_info = MagicMock()
        tool_info.name = "async_ext"
        tool_info.description = "test"
        tool_info.parameters = {}
        tool_info.handler = AsyncMock(return_value="async_result")
        ext_mgr.get_tools.return_value = [tool_info]
        d.register_extension_tools(ext_mgr)
        ctx = _make_ctx()
        result = await d.dispatch("async_ext", {}, ctx)
        assert result == "async_result"

    def test_none_extensions_noop(self):
        d = ToolDispatcher()
        d.register_extension_tools(None)
        assert d.get_definitions() == []
