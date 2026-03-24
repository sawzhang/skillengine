"""
Extension manager - discovery, loading, lifecycle, and event dispatch.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skillengine.adapters.registry import AdapterFactory, AdapterRegistry
from skillengine.events import EventBus
from skillengine.extensions.models import (
    CommandInfo,
    ExtensionHook,
    ExtensionInfo,
    ToolInfo,
)
from skillengine.logging import get_logger

logger = get_logger("extensions")


class ExtensionManager:
    """
    Manages extension discovery, loading, lifecycle, and event dispatch.

    Extensions are discovered from:
    1. Python entry points (group: skillengine.extensions)
    2. Global directory: ~/.skillengine/extensions/*.py
    3. Project-local: ./.skillengine/extensions/*.py

    Each extension module must expose an ``extension(api)`` callable.
    """

    def __init__(
        self,
        engine: Any,
        event_bus: EventBus | None = None,
        adapter_registry: AdapterRegistry | None = None,
    ) -> None:
        from skillengine.engine import SkillsEngine

        assert isinstance(engine, SkillsEngine)
        self.engine: SkillsEngine = engine
        self.event_bus: EventBus | None = event_bus
        self.adapter_registry: AdapterRegistry | None = adapter_registry

        self._extensions: dict[str, ExtensionInfo] = {}
        self._hooks: list[ExtensionHook] = []
        self._commands: list[CommandInfo] = []
        self._tools: list[ToolInfo] = []

    def discover(self) -> list[tuple[str, str, Any]]:
        """
        Find extensions from all sources.

        Returns a list of (name, source, module_or_path) tuples.
        """
        found: list[tuple[str, str, Any]] = []

        # 1. Python entry points
        try:
            if sys.version_info >= (3, 12):
                from importlib.metadata import entry_points

                eps = entry_points(group="skillengine.extensions")
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                if isinstance(all_eps, dict):
                    eps = all_eps.get("skillengine.extensions", [])
                else:
                    eps = all_eps.select(group="skillengine.extensions")

            for ep in eps:
                found.append((ep.name, "entrypoint", ep))
                logger.debug("Discovered entry point extension: %s", ep.name)
        except Exception as e:
            logger.debug("Entry point discovery failed: %s", e)

        # 2. Global directory
        global_dir = Path.home() / ".skillengine" / "extensions"
        found.extend(self._discover_from_dir(global_dir, "global"))

        # 3. Project-local directory
        local_dir = Path.cwd() / ".skillengine" / "extensions"
        found.extend(self._discover_from_dir(local_dir, "local"))

        return found

    def _discover_from_dir(self, directory: Path, source_label: str) -> list[tuple[str, str, Path]]:
        """Discover extensions from a directory."""
        found: list[tuple[str, str, Path]] = []
        if not directory.is_dir():
            return found
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            name = py_file.stem
            found.append((name, source_label, py_file))
            logger.debug("Discovered %s extension: %s (%s)", source_label, name, py_file)
        return found

    def load_extension(self, name: str, source: str, target: Any) -> bool:
        """
        Load a single extension.

        Args:
            name: Extension name
            source: Source label (entrypoint, global, local)
            target: Entry point object or Path to .py file

        Returns:
            True if loaded successfully
        """
        from skillengine.extensions.api import ExtensionAPI

        try:
            # Import the module
            if isinstance(target, Path):
                module = self._import_from_path(name, target)
            else:
                # Entry point - load it
                module = target.load()
                if callable(module):
                    # Entry point resolved to the factory directly
                    factory = module
                    api = ExtensionAPI(self, extension_name=name)
                    info = factory(api)
                    self._extensions[name] = ExtensionInfo(
                        name=name,
                        version=getattr(info, "version", "0.0.0") if info else "0.0.0",
                        description=getattr(info, "description", "") if info else "",
                        source=source,
                    )
                    logger.info("Loaded extension: %s (source=%s)", name, source)
                    return True
                # Otherwise it's a module, fall through

            # Find the extension() callable
            factory = getattr(module, "extension", None)
            if factory is None or not callable(factory):
                logger.warning("Extension %s has no 'extension' callable, skipping", name)
                return False

            api = ExtensionAPI(self, extension_name=name)
            info = factory(api)

            self._extensions[name] = ExtensionInfo(
                name=name,
                version=getattr(info, "version", "0.0.0") if info else "0.0.0",
                description=getattr(info, "description", "") if info else "",
                source=source,
            )
            logger.info("Loaded extension: %s (source=%s)", name, source)
            return True

        except Exception as e:
            logger.warning("Failed to load extension %s: %s", name, e)
            return False

    def _import_from_path(self, name: str, path: Path) -> Any:
        """Import a Python module from a file path."""
        module_name = f"skillengine_ext_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_all(self) -> int:
        """
        Discover and load all extensions.

        Returns:
            Number of extensions loaded
        """
        discovered = self.discover()
        loaded = 0
        for name, source, target in discovered:
            if self.load_extension(name, source, target):
                loaded += 1
        logger.info("Loaded %d/%d extensions", loaded, len(discovered))
        return loaded

    async def emit(self, event: str, **kwargs: Any) -> list[Any]:
        """
        Dispatch an event to registered hooks, sorted by priority.

        Returns:
            List of results from handlers
        """
        relevant = sorted(
            (h for h in self._hooks if h.event == event),
            key=lambda h: h.priority,
        )
        results: list[Any] = []
        for hook in relevant:
            try:
                result = hook.handler(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                logger.warning(
                    "Hook error (event=%s, ext=%s): %s",
                    event,
                    hook.extension_name,
                    e,
                )
        return results

    def get_commands(self) -> list[CommandInfo]:
        """Return all extension-registered commands."""
        return list(self._commands)

    def get_tools(self) -> list[ToolInfo]:
        """Return all extension-registered tools."""
        return list(self._tools)

    def get_extensions(self) -> list[ExtensionInfo]:
        """Return info about all loaded extensions."""
        return list(self._extensions.values())

    def get_extension(self, name: str) -> ExtensionInfo | None:
        """Get info about a specific extension."""
        return self._extensions.get(name)

    # Internal registration methods (called by ExtensionAPI)

    def _register_hook(
        self,
        event: str,
        handler: Callable[..., Any],
        extension_name: str,
        priority: int,
    ) -> None:
        self._hooks.append(
            ExtensionHook(
                event=event,
                handler=handler,
                extension_name=extension_name,
                priority=priority,
            )
        )
        # Also register on the shared EventBus so agent loop events reach extensions
        if self.event_bus is not None:
            self.event_bus.on(event, handler, priority=priority, source=extension_name)

    def _register_command(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        extension_name: str,
        usage: str,
    ) -> None:
        # Check for conflicts
        existing = [c for c in self._commands if c.name == name]
        if existing:
            logger.warning(
                "Command '%s' already registered by %s, overriding with %s",
                name,
                existing[0].extension_name,
                extension_name,
            )
            self._commands = [c for c in self._commands if c.name != name]

        self._commands.append(
            CommandInfo(
                name=name,
                description=description,
                handler=handler,
                source="extension",
                extension_name=extension_name,
                usage=usage or name,
            )
        )

    def _register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
        extension_name: str,
    ) -> None:
        existing = [t for t in self._tools if t.name == name]
        if existing:
            logger.warning(
                "Tool '%s' already registered by %s, overriding with %s",
                name,
                existing[0].extension_name,
                extension_name,
            )
            self._tools = [t for t in self._tools if t.name != name]

        self._tools.append(
            ToolInfo(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
                extension_name=extension_name,
            )
        )

    def _register_adapter(
        self,
        name: str,
        adapter: Any | None = None,
        factory: AdapterFactory | None = None,
        source: str = "",
    ) -> None:
        """Register an LLM adapter from an extension."""
        if self.adapter_registry is None:
            logger.warning(
                "Cannot register adapter '%s': no adapter registry configured",
                name,
            )
            return

        if adapter is not None:
            self.adapter_registry.register(name, adapter, source=source)
        elif factory is not None:
            self.adapter_registry.register_factory(name, factory, source=source)
        else:
            logger.warning(
                "register_adapter('%s') called with neither adapter nor factory",
                name,
            )
