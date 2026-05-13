"""Pool of MCP client connections bridged into the SkillEngine tool dispatcher.

A :class:`MCPConnectionPool` owns a set of live :class:`MCPClient` instances,
each potentially serving multiple tools, and exposes a tidy lifecycle
(``connect_all`` / ``aclose``) suitable for use from :class:`AgentRunner`.

The pool is responsible for:

1. Translating each :class:`MCPServerSpec` into an active client.
2. Listing tools per server and giving them a stable namespaced name
   (``<server>__<tool>``) so multiple servers can co-exist.
3. Registering every tool with a SkillEngine
   :class:`~skillengine.tools.dispatcher.ToolDispatcher` (or compatible
   object — anything with ``register(name, handler, definition)``).
4. Unregistering tools and closing transports on teardown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from .client import MCPClient
from .spec import MCPServerSpec, coerce_spec
from .transport import StdioTransport, Transport

logger = logging.getLogger(__name__)


@dataclass
class _Connection:
    spec: MCPServerSpec
    client: MCPClient
    name: str  # logical/prefix name used for tool namespacing
    tool_names: list[str] = field(default_factory=list)


class MCPConnectionPool:
    """Manage a set of MCP client connections and bridge their tools.

    The class is transport-agnostic for tests: callers may pass a
    ``transport_factory`` that returns a :class:`Transport` for a given
    :class:`MCPServerSpec`. By default it spawns stdio subprocesses.
    """

    def __init__(
        self,
        *,
        transport_factory: Any = None,
    ) -> None:
        self._connections: dict[str, _Connection] = {}
        self._transport_factory = transport_factory or _default_transport_factory

    # ---- lifecycle -------------------------------------------------------

    async def connect(
        self,
        spec: MCPServerSpec | str | dict[str, Any],
        *,
        name: str | None = None,
    ) -> MCPClient:
        """Connect to one MCP server and return its live client."""
        spec_obj = coerce_spec(spec)
        logical = name or spec_obj.name or _default_name(spec_obj, taken=self._connections.keys())
        if logical in self._connections:
            raise ValueError(f"MCP server name already in use: {logical!r}")
        transport = await self._transport_factory(spec_obj)
        client = MCPClient(transport)
        await client.start()
        # Pre-fetch tool list so a follow-up ``register_with_dispatcher`` call is cheap.
        await client.list_tools()
        self._connections[logical] = _Connection(spec=spec_obj, client=client, name=logical)
        return client

    async def connect_all(
        self,
        specs: list[MCPServerSpec | str | dict[str, Any]],
    ) -> list[MCPClient]:
        """Connect to multiple servers; failures are aggregated into one error."""
        clients: list[MCPClient] = []
        errors: list[tuple[Any, BaseException]] = []
        for spec in specs:
            try:
                clients.append(await self.connect(spec))
            except BaseException as exc:  # noqa: BLE001 — aggregate and re-raise
                errors.append((spec, exc))
        if errors:
            messages = "; ".join(f"{s!r}: {e}" for s, e in errors)
            await self.aclose()
            raise RuntimeError(f"Failed to connect MCP servers: {messages}")
        return clients

    async def aclose(self) -> None:
        """Disconnect every client and clear state."""
        # Disconnect concurrently but tolerate individual failures.
        await asyncio.gather(
            *(conn.client.aclose() for conn in self._connections.values()),
            return_exceptions=True,
        )
        self._connections.clear()

    async def __aenter__(self) -> MCPConnectionPool:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # ---- introspection ---------------------------------------------------

    def list_connections(self) -> list[str]:
        return list(self._connections.keys())

    def get_client(self, name: str) -> MCPClient | None:
        conn = self._connections.get(name)
        return conn.client if conn else None

    def get_tool_names(self, name: str) -> list[str]:
        conn = self._connections.get(name)
        return list(conn.tool_names) if conn else []

    # ---- dispatcher bridge ----------------------------------------------

    def register_with_dispatcher(self, dispatcher: Any) -> int:
        """Register every pool tool with a SkillEngine ToolDispatcher-like object.

        Returns the number of tools registered. Names are prefixed with the
        server's logical name (``<server>__<tool>``) to avoid collisions.

        The dispatcher must expose ``register(name, handler, definition)`` —
        which matches both :class:`skillengine.tools.dispatcher.ToolDispatcher`
        and :class:`skillengine.tools.registry.ToolRegistry`-compatible types.
        """
        count = 0
        for logical, conn in self._connections.items():
            defs = conn.client.tool_definitions(prefix=logical)
            conn.tool_names = []
            for td in defs:
                definition = {
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters,
                    },
                }
                handler = _wrap_handler(td.handler)
                dispatcher.register(td.name, handler, definition)
                conn.tool_names.append(td.name)
                count += 1
        return count

    def unregister_from_dispatcher(self, dispatcher: Any) -> int:
        """Remove every previously-registered tool from ``dispatcher``."""
        count = 0
        for conn in self._connections.values():
            for tool_name in conn.tool_names:
                with suppress(Exception):
                    if dispatcher.unregister(tool_name):
                        count += 1
            conn.tool_names = []
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _default_transport_factory(spec: MCPServerSpec) -> Transport:
    return await StdioTransport.spawn(spec.to_stdio_spec())


def _wrap_handler(handler: Any) -> Any:
    """Adapt an MCP ``handler(args) -> str`` to a dispatcher tool handler.

    The :class:`~skillengine.tools.dispatcher.ToolDispatcher` invokes its
    handlers as ``handler(name, args, ctx, on_output)``. MCP tool handlers
    only care about ``args`` and return a string, so we ignore the extras.
    """

    async def _call(_name: Any, args: dict[str, Any], _ctx: Any, _on_output: Any = None) -> str:
        return await handler(args)

    return _call


def _default_name(spec: MCPServerSpec, *, taken: Any) -> str:
    """Pick a unique-ish logical name from the spec's command."""
    base = (spec.command or "mcp").rsplit("/", 1)[-1] or "mcp"
    taken_set = set(taken)
    if base not in taken_set:
        return base
    i = 2
    while f"{base}{i}" in taken_set:
        i += 1
    return f"{base}{i}"


__all__ = ["MCPConnectionPool"]
