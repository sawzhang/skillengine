"""
Dynamic adapter registry for runtime LLM provider management.

Provides a central registry where adapters can be registered, looked up,
and switched at runtime. Extensions can register custom providers.

Example:
    from skillengine.adapters.registry import AdapterRegistry

    registry = AdapterRegistry()

    # Register by instance
    registry.register("openai", my_openai_adapter)

    # Register by factory (lazy creation)
    registry.register_factory("anthropic", lambda engine: AnthropicAdapter(engine))

    # Look up
    adapter = registry.get("openai")

    # Switch at runtime
    registry.set_default("anthropic")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from skillengine.logging import get_logger

if TYPE_CHECKING:
    from skillengine.adapters.base import LLMAdapter
    from skillengine.engine import SkillsEngine

logger = get_logger("adapters.registry")

# Type for adapter factory functions: (engine) -> LLMAdapter
AdapterFactory = Callable[["SkillsEngine"], "LLMAdapter"]


class _AdapterEntry:
    """Internal entry holding either an adapter instance or a factory."""

    __slots__ = ("name", "instance", "factory", "source")

    def __init__(
        self,
        name: str,
        instance: LLMAdapter | None = None,
        factory: AdapterFactory | None = None,
        source: str = "",
    ) -> None:
        self.name = name
        self.instance = instance
        self.factory = factory
        self.source = source


class AdapterRegistry:
    """
    A dynamic registry of LLM adapters.

    Adapters can be registered by instance (eager) or by factory (lazy).
    A default adapter name can be set for convenience.

    Thread Safety:
        The registry is designed for single-threaded async usage.
        Registration and lookup are not protected by locks.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _AdapterEntry] = {}
        self._default_name: str | None = None

    def register(
        self,
        name: str,
        adapter: LLMAdapter,
        source: str = "",
    ) -> None:
        """
        Register an adapter instance.

        Args:
            name: Unique adapter name (e.g., "openai", "my-local-llm")
            adapter: LLMAdapter instance
            source: Who registered it (extension name, etc.)

        Raises:
            ValueError: If name is empty
        """
        if not name:
            raise ValueError("Adapter name must not be empty")

        if name in self._entries:
            logger.debug("Overriding adapter: %s", name)

        self._entries[name] = _AdapterEntry(
            name=name,
            instance=adapter,
            source=source,
        )
        logger.debug("Registered adapter: %s (source=%s)", name, source or "manual")

        # Auto-set default if this is the first entry
        if self._default_name is None:
            self._default_name = name

    def register_factory(
        self,
        name: str,
        factory: AdapterFactory,
        source: str = "",
    ) -> None:
        """
        Register an adapter factory for lazy creation.

        The factory is called with a ``SkillsEngine`` the first time
        the adapter is requested via ``get()``.

        Args:
            name: Unique adapter name
            factory: Callable that creates an LLMAdapter given a SkillsEngine
            source: Who registered it (extension name, etc.)

        Raises:
            ValueError: If name is empty
        """
        if not name:
            raise ValueError("Adapter name must not be empty")

        if name in self._entries:
            logger.debug("Overriding adapter factory: %s", name)

        self._entries[name] = _AdapterEntry(
            name=name,
            factory=factory,
            source=source,
        )
        logger.debug("Registered adapter factory: %s (source=%s)", name, source or "manual")

        if self._default_name is None:
            self._default_name = name

    def get(self, name: str, engine: SkillsEngine | None = None) -> LLMAdapter:
        """
        Get an adapter by name.

        If the adapter was registered via factory and hasn't been created yet,
        the factory is called with ``engine`` to create it.

        Args:
            name: Adapter name
            engine: SkillsEngine for factory creation (required for factories)

        Returns:
            The LLMAdapter instance

        Raises:
            KeyError: If adapter not found
            RuntimeError: If factory needs engine but none provided
        """
        entry = self._entries.get(name)
        if entry is None:
            available = ", ".join(self._entries.keys()) or "(none)"
            raise KeyError(f"Adapter '{name}' not found. Available: {available}")

        if entry.instance is not None:
            return entry.instance

        # Lazy factory creation
        if entry.factory is not None:
            if engine is None:
                raise RuntimeError(f"Adapter '{name}' requires a SkillsEngine for factory creation")
            logger.debug("Creating adapter '%s' from factory", name)
            entry.instance = entry.factory(engine)
            return entry.instance

        raise RuntimeError(f"Adapter '{name}' has no instance or factory")

    def get_default(self, engine: SkillsEngine | None = None) -> LLMAdapter | None:
        """
        Get the default adapter.

        Returns:
            The default LLMAdapter, or None if no adapters registered
        """
        if self._default_name is None:
            return None
        try:
            return self.get(self._default_name, engine=engine)
        except KeyError:
            return None

    @property
    def default_name(self) -> str | None:
        """Get the name of the default adapter."""
        return self._default_name

    def set_default(self, name: str) -> None:
        """
        Set the default adapter.

        Args:
            name: Adapter name (must already be registered)

        Raises:
            KeyError: If adapter not found
        """
        if name not in self._entries:
            available = ", ".join(self._entries.keys()) or "(none)"
            raise KeyError(f"Adapter '{name}' not found. Available: {available}")
        self._default_name = name
        logger.debug("Default adapter set to: %s", name)

    def unregister(self, name: str) -> bool:
        """
        Remove a registered adapter.

        Args:
            name: Adapter name

        Returns:
            True if removed, False if not found
        """
        if name not in self._entries:
            return False

        del self._entries[name]
        logger.debug("Unregistered adapter: %s", name)

        # Clear default if it was the removed adapter
        if self._default_name == name:
            self._default_name = next(iter(self._entries), None)

        return True

    def unregister_by_source(self, source: str) -> int:
        """
        Remove all adapters registered by a given source.

        Args:
            source: Source identifier (e.g., extension name)

        Returns:
            Number of adapters removed
        """
        to_remove = [name for name, entry in self._entries.items() if entry.source == source]
        for name in to_remove:
            self.unregister(name)
        return len(to_remove)

    def has(self, name: str) -> bool:
        """Check if an adapter is registered."""
        return name in self._entries

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._entries.keys())

    def list_by_source(self, source: str) -> list[str]:
        """List adapter names registered by a given source."""
        return [name for name, entry in self._entries.items() if entry.source == source]

    def get_info(self, name: str) -> dict[str, Any]:
        """
        Get info about a registered adapter.

        Returns:
            Dict with name, source, has_instance, has_factory
        """
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Adapter '{name}' not found")

        return {
            "name": entry.name,
            "source": entry.source,
            "has_instance": entry.instance is not None,
            "has_factory": entry.factory is not None,
            "is_default": entry.name == self._default_name,
        }

    def clear(self) -> None:
        """Remove all registered adapters."""
        self._entries.clear()
        self._default_name = None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        names = ", ".join(self._entries.keys())
        default = f" default={self._default_name}" if self._default_name else ""
        return f"AdapterRegistry([{names}]{default})"
