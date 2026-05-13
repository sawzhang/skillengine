"""MCP server — expose SkillEngine skills and tools to MCP clients.

This is the counterpart of :mod:`skillengine.mcp.client`. It speaks the same
JSON-RPC over the same transports, but in the *server* role: handles
``initialize``, ``tools/list``, ``tools/call`` and routes calls to handlers
registered by the host application.

Typical use::

    from skillengine import SkillsEngine
    from skillengine.mcp import StdioTransport
    from skillengine.mcp.server import MCPServer, serve_stdio

    engine = SkillsEngine(...)
    await serve_stdio(engine=engine, name="my-skills", version="1.0")

Or programmatically with a custom transport (tests, in-process bridge)::

    server = MCPServer(transport, name="srv")
    server.register_engine_skills(engine)
    await server.serve()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..engine import SkillsEngine
from ..models import Skill
from ..tools.registry import ToolDefinition
from .protocol import MCP_PROTOCOL_VERSION, ToolCallResult
from .transport import StdioServerSpec, StdioTransport, Transport, TransportClosed

logger = logging.getLogger(__name__)


ServerToolHandler = Callable[[dict[str, Any]], Awaitable["str | ToolCallResult | dict[str, Any]"]]


@dataclass
class ServerTool:
    """A tool exposed by an :class:`MCPServer`.

    Handlers may return any of:

    - ``str`` — wrapped as ``{type: text, text: ...}`` content
    - :class:`ToolCallResult` — returned as-is
    - ``dict`` — assumed to already be a valid ``tools/call`` result payload
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ServerToolHandler


# ---------------------------------------------------------------------------
# Conversions: SkillEngine -> ServerTool
# ---------------------------------------------------------------------------


_SKILL_TOOL_PREFIX = "skill__"


def skill_as_server_tool(skill: Skill) -> ServerTool:
    """Wrap a :class:`Skill` so MCP clients can invoke it as a tool.

    The remote tool name is ``skill__<skill.name>``. Calling it returns the
    rendered skill content (with ``$ARGUMENTS`` substituted) — the LLM client
    can then either incorporate that content into its next turn or pass it
    through as instructions, mirroring SkillEngine's own on-demand-load flow.
    """

    async def _handler(args: dict[str, Any]) -> str:
        arguments = str(args.get("arguments", "") or "")
        return _render_skill(skill, arguments)

    return ServerTool(
        name=f"{_SKILL_TOOL_PREFIX}{skill.name}",
        description=skill.description or f"SkillEngine skill: {skill.name}",
        input_schema={
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "string",
                    "description": skill.argument_hint or "Arguments forwarded to the skill",
                },
            },
            "required": [],
        },
        handler=_handler,
    )


def _render_skill(skill: Skill, arguments: str) -> str:
    """Naive ``$ARGUMENTS`` substitution.

    We deliberately keep this conservative: the full SkillEngine substitution
    pipeline (``!`cmd``, ``$1..$N``, ``${CLAUDE_SESSION_ID}``) runs inside
    :class:`AgentRunner`. An MCP client driving SkillEngine externally may not
    have a session/shell context, so we only substitute ``$ARGUMENTS`` here.
    """
    return skill.content.replace("$ARGUMENTS", arguments)


def tool_definition_as_server_tool(definition: ToolDefinition) -> ServerTool:
    """Wrap a SkillEngine :class:`ToolDefinition` for MCP exposure."""

    handler = definition.handler

    async def _handler(args: dict[str, Any]) -> str:
        if handler is None:
            raise RuntimeError(f"tool {definition.name!r} has no handler")
        result = await handler(args)
        if isinstance(result, str):
            return result
        return str(result)

    return ServerTool(
        name=definition.name,
        description=definition.description,
        input_schema=definition.parameters or {"type": "object", "properties": {}, "required": []},
        handler=_handler,
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MCPServer:
    """Minimal MCP server bound to a single :class:`Transport`."""

    def __init__(
        self,
        transport: Transport,
        *,
        name: str = "skillengine",
        version: str = "0.3",
        instructions: str | None = None,
    ) -> None:
        self._transport = transport
        self._name = name
        self._version = version
        self._instructions = instructions
        self._tools: dict[str, ServerTool] = {}
        self._initialized = asyncio.Event()
        self._stopped = asyncio.Event()
        self._serve_task: asyncio.Task[None] | None = None

    # ---- registration ----------------------------------------------------

    def register_tool(self, tool: ServerTool) -> None:
        self._tools[tool.name] = tool

    def register_skill(self, skill: Skill) -> None:
        self.register_tool(skill_as_server_tool(skill))

    def register_tool_definition(self, definition: ToolDefinition) -> None:
        self.register_tool(tool_definition_as_server_tool(definition))

    def register_engine_skills(self, engine: SkillsEngine) -> int:
        """Register every eligible skill from ``engine``. Returns count added."""
        snapshot = engine.get_snapshot()
        count = 0
        for skill in snapshot.skills:
            self.register_skill(skill)
            count += 1
        return count

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    # ---- loop ------------------------------------------------------------

    async def serve(self) -> None:
        """Run until the transport closes."""
        try:
            while True:
                try:
                    message = await self._transport.receive()
                except TransportClosed:
                    break
                await self._handle(message)
        finally:
            self._stopped.set()

    def start(self) -> asyncio.Task[None]:
        if self._serve_task is None or self._serve_task.done():
            self._serve_task = asyncio.create_task(self.serve(), name="mcp-server")
        return self._serve_task

    async def aclose(self) -> None:
        try:
            await self._transport.close()
        except Exception:
            pass
        if self._serve_task is not None and not self._serve_task.done():
            try:
                await asyncio.wait_for(self._serve_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._serve_task.cancel()
                try:
                    await self._serve_task
                except (asyncio.CancelledError, Exception):
                    pass

    # ---- dispatch --------------------------------------------------------

    async def _handle(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            await self._reply(
                msg_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "serverInfo": {"name": self._name, "version": self._version},
                    "capabilities": {"tools": {"listChanged": False}},
                    **({"instructions": self._instructions} if self._instructions else {}),
                },
            )
            return

        if method == "notifications/initialized":
            self._initialized.set()
            return

        # All other methods require a completed handshake. Most clients send
        # ``initialized`` before any further request, but be permissive: only
        # *reject* non-handshake calls when we explicitly know we haven't been
        # initialized — and only do so for requests (which carry an id).
        if method == "tools/list":
            await self._reply(msg_id, {"tools": self._serialize_tools()})
            return

        if method == "tools/call":
            await self._handle_tool_call(msg_id, params)
            return

        if method == "ping":
            await self._reply(msg_id, {})
            return

        # Unknown method
        if msg_id is not None:
            await self._error(msg_id, code=-32601, message=f"Method not found: {method}")

    async def _handle_tool_call(self, msg_id: Any, params: dict[str, Any]) -> None:
        name = params.get("name")
        if not isinstance(name, str):
            await self._error(msg_id, code=-32602, message="tools/call missing 'name'")
            return
        tool = self._tools.get(name)
        if tool is None:
            await self._error(msg_id, code=-32601, message=f"Unknown tool: {name}")
            return
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            await self._error(msg_id, code=-32602, message="'arguments' must be an object")
            return
        try:
            raw = await tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 — translate every error into isError result
            logger.exception("tool %s handler raised", name)
            await self._reply(
                msg_id,
                {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                },
            )
            return
        await self._reply(msg_id, self._normalize_tool_result(raw))

    @staticmethod
    def _normalize_tool_result(raw: Any) -> dict[str, Any]:
        if isinstance(raw, ToolCallResult):
            return {"content": raw.content, "isError": raw.is_error}
        if isinstance(raw, dict) and "content" in raw:
            payload = {"content": list(raw["content"]), "isError": bool(raw.get("isError", False))}
            return payload
        return {"content": [{"type": "text", "text": str(raw)}], "isError": False}

    def _serialize_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema or {"type": "object", "properties": {}},
            }
            for t in self._tools.values()
        ]

    # ---- low-level send --------------------------------------------------

    async def _reply(self, msg_id: Any, result: dict[str, Any]) -> None:
        if msg_id is None:
            return
        await self._transport.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    async def _error(self, msg_id: Any, *, code: int, message: str) -> None:
        if msg_id is None:
            return
        await self._transport.send(
            {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
        )


# ---------------------------------------------------------------------------
# Convenience launchers
# ---------------------------------------------------------------------------


async def serve_stdio(
    *,
    engine: SkillsEngine | None = None,
    tools: list[ServerTool] | None = None,
    name: str = "skillengine",
    version: str = "0.3",
    instructions: str | None = None,
) -> None:
    """Run an :class:`MCPServer` over the current process's stdin/stdout.

    This is the entry point used by ``python -m skillengine.mcp`` so other
    agents (Claude Desktop, Cursor, MCP Inspector) can discover SkillEngine
    skills via the standard stdio transport.
    """
    transport = await _stdio_transport_for_self()
    server = MCPServer(transport, name=name, version=version, instructions=instructions)
    if engine is not None:
        server.register_engine_skills(engine)
    for tool in tools or []:
        server.register_tool(tool)
    await server.serve()


async def _stdio_transport_for_self() -> Transport:
    """Build a :class:`StdioTransport` reading our stdin / writing our stdout.

    :func:`StdioTransport.spawn` is for the *client* role (spawn a child).
    On the server side we are the spawned child, so we wrap our own pipes
    using a thin adapter.
    """
    # Wrap sys.stdin / sys.stdout into asyncio streams.
    loop = asyncio.get_event_loop()

    reader = asyncio.StreamReader(loop=loop)
    reader_proto = asyncio.StreamReaderProtocol(reader, loop=loop)
    await loop.connect_read_pipe(lambda: reader_proto, sys.stdin)

    write_transport, write_proto = await loop.connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(loop=loop), sys.stdout
    )
    writer = asyncio.StreamWriter(write_transport, write_proto, None, loop)

    return _StreamPipeTransport(reader, writer)


@dataclass
class _StreamPipeTransport(Transport):
    """Adapter that exposes a reader/writer pair as a :class:`Transport`."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    _closed: bool = field(default=False, init=False)

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise TransportClosed("stdio pipe transport closed")
        import json

        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self.writer.write(payload)
        await self.writer.drain()

    async def receive(self) -> dict[str, Any]:
        if self._closed:
            raise TransportClosed("stdio pipe transport closed")
        line = await self.reader.readline()
        if not line:
            self._closed = True
            raise TransportClosed("stdin EOF")
        import json

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportClosed(f"non-JSON line on stdin: {line!r}") from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.writer.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed


__all__ = [
    "MCPServer",
    "ServerTool",
    "ServerToolHandler",
    "StdioServerSpec",  # re-exported for convenience
    "StdioTransport",
    "serve_stdio",
    "skill_as_server_tool",
    "tool_definition_as_server_tool",
]
