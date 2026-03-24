"""Tests for the extension system."""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from skillengine import SkillsConfig, SkillsEngine
from skillengine.extensions.api import ExtensionAPI
from skillengine.extensions.manager import ExtensionManager
from skillengine.extensions.models import (
    SKILL_LOADED,
    SESSION_START,
    CommandInfo,
    ExtensionHook,
    ExtensionInfo,
    ToolInfo,
)


@pytest.fixture
def engine() -> SkillsEngine:
    config = SkillsConfig(skill_dirs=[])
    return SkillsEngine(config=config)


@pytest.fixture
def manager(engine: SkillsEngine) -> ExtensionManager:
    return ExtensionManager(engine)


class TestExtensionModels:
    def test_extension_info_defaults(self) -> None:
        info = ExtensionInfo(name="test")
        assert info.name == "test"
        assert info.version == "0.0.0"
        assert info.description == ""

    def test_command_info(self) -> None:
        cmd = CommandInfo(name="hello", description="Say hello", source="extension")
        assert cmd.name == "hello"
        assert cmd.source == "extension"

    def test_tool_info(self) -> None:
        tool = ToolInfo(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "search"
        assert tool.parameters["type"] == "object"

    def test_extension_hook(self) -> None:
        hook = ExtensionHook(
            event=SESSION_START,
            handler=lambda: None,
            extension_name="test",
            priority=5,
        )
        assert hook.event == SESSION_START
        assert hook.priority == 5


class TestExtensionAPI:
    def test_on_registers_hook(self, manager: ExtensionManager) -> None:
        api = ExtensionAPI(manager, extension_name="test-ext")
        handler = lambda: None
        api.on("session_start", handler, priority=10)
        assert len(manager._hooks) == 1
        assert manager._hooks[0].event == "session_start"
        assert manager._hooks[0].priority == 10
        assert manager._hooks[0].extension_name == "test-ext"

    def test_register_command(self, manager: ExtensionManager) -> None:
        api = ExtensionAPI(manager, extension_name="test-ext")
        handler = lambda args: "hello"
        api.register_command("hello", handler, "Say hello", "/hello [name]")
        assert len(manager._commands) == 1
        assert manager._commands[0].name == "hello"

    def test_register_tool(self, manager: ExtensionManager) -> None:
        api = ExtensionAPI(manager, extension_name="test-ext")
        handler = lambda args: "result"
        api.register_tool("search", handler, "Search", {"type": "object"})
        assert len(manager._tools) == 1
        assert manager._tools[0].name == "search"

    def test_engine_property(self, manager: ExtensionManager, engine: SkillsEngine) -> None:
        api = ExtensionAPI(manager, extension_name="test-ext")
        assert api.engine is engine

    def test_config_property(self, manager: ExtensionManager, engine: SkillsEngine) -> None:
        api = ExtensionAPI(manager, extension_name="test-ext")
        assert api.config is engine.config


class TestExtensionManager:
    def test_discover_from_dir(self, manager: ExtensionManager, tmp_path: Path) -> None:
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "my_ext.py").write_text(
            dedent("""
            def extension(api):
                api.register_command("greet", lambda args: "hi", "Greet")
            """).strip()
        )
        (ext_dir / "_private.py").write_text("# ignored")

        found = manager._discover_from_dir(ext_dir, "test")
        assert len(found) == 1
        assert found[0][0] == "my_ext"

    def test_discover_nonexistent_dir(self, manager: ExtensionManager) -> None:
        found = manager._discover_from_dir(Path("/nonexistent"), "test")
        assert found == []

    def test_load_extension_from_file(
        self, manager: ExtensionManager, tmp_path: Path
    ) -> None:
        ext_file = tmp_path / "my_ext.py"
        ext_file.write_text(
            dedent("""
            def extension(api):
                api.register_command("greet", lambda args: "hi", "Greet")
                api.register_tool("search", lambda args: "found", "Search", {"type": "object"})
                api.on("session_start", lambda: None)
            """).strip()
        )
        assert manager.load_extension("my_ext", "test", ext_file)
        assert "my_ext" in manager._extensions
        assert len(manager.get_commands()) == 1
        assert len(manager.get_tools()) == 1
        assert len(manager._hooks) == 1

    def test_load_extension_missing_callable(
        self, manager: ExtensionManager, tmp_path: Path
    ) -> None:
        ext_file = tmp_path / "bad_ext.py"
        ext_file.write_text("x = 1")
        assert not manager.load_extension("bad_ext", "test", ext_file)

    def test_load_extension_with_error(
        self, manager: ExtensionManager, tmp_path: Path
    ) -> None:
        ext_file = tmp_path / "err_ext.py"
        ext_file.write_text(
            dedent("""
            def extension(api):
                raise RuntimeError("fail")
            """).strip()
        )
        assert not manager.load_extension("err_ext", "test", ext_file)

    def test_event_emission(self, manager: ExtensionManager) -> None:
        results: list[str] = []

        def handler_a(**kwargs: object) -> str:
            results.append("a")
            return "a"

        def handler_b(**kwargs: object) -> str:
            results.append("b")
            return "b"

        # Register with priority (lower first)
        manager._register_hook(SKILL_LOADED, handler_b, "ext-b", priority=10)
        manager._register_hook(SKILL_LOADED, handler_a, "ext-a", priority=1)

        emit_results = asyncio.get_event_loop().run_until_complete(
            manager.emit(SKILL_LOADED, name="test")
        )
        assert results == ["a", "b"]  # a has lower priority, runs first
        assert emit_results == ["a", "b"]

    def test_event_emission_async(self, manager: ExtensionManager) -> None:
        async def async_handler(**kwargs: object) -> str:
            return "async_result"

        manager._register_hook("test_event", async_handler, "ext", priority=0)
        results = asyncio.get_event_loop().run_until_complete(
            manager.emit("test_event")
        )
        assert results == ["async_result"]

    def test_command_conflict_detection(self, manager: ExtensionManager) -> None:
        manager._register_command("hello", lambda a: "1", "First", "ext1", "")
        manager._register_command("hello", lambda a: "2", "Second", "ext2", "")
        # Later wins
        assert len(manager.get_commands()) == 1
        assert manager.get_commands()[0].extension_name == "ext2"

    def test_tool_conflict_detection(self, manager: ExtensionManager) -> None:
        manager._register_tool("search", lambda a: "1", "First", {}, "ext1")
        manager._register_tool("search", lambda a: "2", "Second", {}, "ext2")
        assert len(manager.get_tools()) == 1
        assert manager.get_tools()[0].extension_name == "ext2"

    def test_get_extensions(self, manager: ExtensionManager) -> None:
        manager._extensions["foo"] = ExtensionInfo(name="foo", version="1.0")
        exts = manager.get_extensions()
        assert len(exts) == 1
        assert exts[0].name == "foo"

    def test_get_extension(self, manager: ExtensionManager) -> None:
        manager._extensions["foo"] = ExtensionInfo(name="foo")
        assert manager.get_extension("foo") is not None
        assert manager.get_extension("bar") is None

    def test_discover_entry_points(self, manager: ExtensionManager) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "test_ep"
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            found = manager.discover()
            ep_entries = [f for f in found if f[1] == "entrypoint"]
            assert len(ep_entries) == 1
            assert ep_entries[0][0] == "test_ep"


class TestEngineExtensions:
    def test_init_extensions(self) -> None:
        config = SkillsConfig(skill_dirs=[])
        engine = SkillsEngine(config=config)
        assert engine.extensions is None
        ext = engine.init_extensions()
        assert engine.extensions is ext
        assert isinstance(ext, ExtensionManager)
