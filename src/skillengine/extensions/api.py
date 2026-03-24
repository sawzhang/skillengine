"""
Extension API - the registration surface passed to extension factory functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skillengine.adapters.base import LLMAdapter
    from skillengine.adapters.registry import AdapterFactory
    from skillengine.config import SkillsConfig
    from skillengine.engine import SkillsEngine
    from skillengine.extensions.manager import ExtensionManager


class ExtensionAPI:
    """
    API object passed to extension factory functions.

    Extensions use this to register commands, tools, and event hooks.

    Example extension:
        def extension(api: ExtensionAPI):
            api.on("session_start", lambda: print("Session started"))
            api.register_command("hello", lambda args: "Hello!", "Say hello")
    """

    def __init__(self, manager: ExtensionManager, extension_name: str = "") -> None:
        self._manager = manager
        self._extension_name = extension_name

    @property
    def engine(self) -> SkillsEngine:
        """Access the SkillsEngine instance."""
        return self._manager.engine

    @property
    def config(self) -> SkillsConfig:
        """Access the SkillsConfig."""
        return self._manager.engine.config

    def on(self, event: str, handler: Callable[..., Any], priority: int = 0) -> None:
        """Register an event hook."""
        self._manager._register_hook(event, handler, self._extension_name, priority)

    def register_command(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        usage: str = "",
    ) -> None:
        """Register a slash command."""
        self._manager._register_command(name, handler, description, self._extension_name, usage)

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool for LLM function calling."""
        self._manager._register_tool(
            name, handler, description, parameters or {}, self._extension_name
        )

    def register_adapter(
        self,
        name: str,
        adapter: LLMAdapter | None = None,
        factory: AdapterFactory | None = None,
    ) -> None:
        """
        Register an LLM adapter for the agent to use.

        Extensions can register custom LLM providers at runtime. The adapter
        can then be activated via ``AgentRunner.set_adapter(name)``.

        Either ``adapter`` (an instance) or ``factory`` (a callable) must
        be provided.

        Args:
            name: Unique adapter name (e.g., "my-local-llm")
            adapter: An LLMAdapter instance (eager registration)
            factory: A callable ``(engine) -> LLMAdapter`` (lazy creation)

        Example::

            def my_extension(api: ExtensionAPI):
                api.register_adapter(
                    "my-local-llm",
                    factory=lambda engine: OpenAIAdapter(
                        engine, base_url="http://localhost:8080/v1"
                    ),
                )
        """
        self._manager._register_adapter(
            name,
            adapter=adapter,
            factory=factory,
            source=self._extension_name,
        )
