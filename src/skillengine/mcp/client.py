"""High-level MCP client.

Manages the JSON-RPC request/response correlation, handshake, and tool listing
on top of a :class:`Transport`. It also adapts MCP tools to SkillEngine's
:class:`~skillengine.tools.registry.ToolDefinition` shape so MCP-hosted tools
can be plugged directly into an :class:`~skillengine.agent.AgentRunner`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from ..tools.registry import ToolDefinition
from .protocol import (
    MCP_PROTOCOL_VERSION,
    InitializeResult,
    JSONRPCError,
    MCPTool,
    ToolCallResult,
    make_notification,
    make_request,
)
from .transport import StdioServerSpec, StdioTransport, Transport, TransportClosed

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """Generic MCP client error (transport failure, protocol violation)."""


class MCPToolError(RuntimeError):
    """Raised when ``tools/call`` returns ``isError: true``."""

    def __init__(self, tool_name: str, result: ToolCallResult) -> None:
        super().__init__(f"MCP tool {tool_name!r} returned an error: {result.text() or result.raw}")
        self.tool_name = tool_name
        self.result = result


_DEFAULT_TIMEOUT = 30.0


class MCPClient:
    """Async MCP client.

    Lifecycle::

        client = MCPClient(transport)
        await client.start()                    # spawn reader + initialize
        tools = await client.list_tools()
        result = await client.call_tool("echo", {"text": "hi"})
        await client.aclose()

    Or use the async-context-manager helpers :meth:`connect_stdio` /
    :meth:`connect`::

        async with MCPClient.connect_stdio(spec) as client:
            ...
    """

    def __init__(
        self,
        transport: Transport,
        *,
        client_name: str = "skillengine",
        client_version: str = "0.3",
        default_timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._transport = transport
        self._client_name = client_name
        self._client_version = client_version
        self._default_timeout = default_timeout

        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._server_info: InitializeResult | None = None
        self._tools_cache: list[MCPTool] | None = None

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    @contextlib.asynccontextmanager
    async def connect_stdio(
        cls,
        spec: StdioServerSpec,
        **kwargs: Any,
    ):
        """Spawn an MCP server subprocess and yield a started client."""
        transport = await StdioTransport.spawn(spec)
        client = cls(transport, **kwargs)
        try:
            await client.start()
            yield client
        finally:
            await client.aclose()

    @classmethod
    @contextlib.asynccontextmanager
    async def connect(cls, transport: Transport, **kwargs: Any):
        """Wrap an arbitrary transport in a started client (test-friendly)."""
        client = cls(transport, **kwargs)
        try:
            await client.start()
            yield client
        finally:
            await client.aclose()

    async def start(self) -> InitializeResult:
        """Spawn the reader task, perform handshake, return server info."""
        if self._started:
            assert self._server_info is not None
            return self._server_info
        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="mcp-client-reader")
        self._server_info = await self._initialize()
        return self._server_info

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel any in-flight requests so callers don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPClientError("MCP client closed before response arrived"))
        self._pending.clear()
        with contextlib.suppress(Exception):
            await self._transport.close()
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # ---- public RPC surface ----------------------------------------------

    @property
    def server_info(self) -> InitializeResult | None:
        return self._server_info

    async def list_tools(self, *, force_refresh: bool = False) -> list[MCPTool]:
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
        data = await self._request("tools/list", {})
        tools = [MCPTool.from_dict(t) for t in data.get("tools", [])]
        self._tools_cache = tools
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        raise_on_error: bool = False,
        timeout: float | None = None,
    ) -> ToolCallResult:
        data = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        result = ToolCallResult.from_dict(data)
        if result.is_error and raise_on_error:
            raise MCPToolError(name, result)
        return result

    # ---- SkillEngine adapter ---------------------------------------------

    def tool_definitions(self, *, prefix: str | None = None) -> list[ToolDefinition]:
        """Return MCP tools wrapped as SkillEngine :class:`ToolDefinition` objects.

        :param prefix: optional ``"<prefix>__"`` prepended to each tool name to
            avoid collisions when multiple MCP servers expose tools with the
            same name (e.g. ``everything__echo`` and ``github__echo``).
        """
        if self._tools_cache is None:
            raise MCPClientError("call list_tools() before tool_definitions()")
        defs: list[ToolDefinition] = []
        for tool in self._tools_cache:
            local_name = f"{prefix}__{tool.name}" if prefix else tool.name
            defs.append(
                ToolDefinition(
                    name=local_name,
                    description=tool.description or f"MCP tool {tool.name}",
                    parameters=tool.input_schema or {"type": "object", "properties": {}},
                    handler=self._make_handler(tool.name),
                )
            )
        return defs

    def _make_handler(self, remote_name: str):
        async def _handler(args: dict[str, Any]) -> str:
            result = await self.call_tool(remote_name, args)
            text = result.text()
            if result.is_error:
                # Surface the error as readable text — agents expect strings here.
                return f"[mcp-error] {text}"
            return text

        return _handler

    # ---- internals --------------------------------------------------------

    async def _initialize(self) -> InitializeResult:
        data = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self._client_name, "version": self._client_version},
            },
        )
        await self._transport.send(make_notification("notifications/initialized", {}))
        return InitializeResult.from_dict(data)

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise MCPClientError("client is closed")
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self._transport.send(make_request(method, request_id, params))
        except Exception:
            self._pending.pop(request_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout=timeout or self._default_timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise MCPClientError(f"MCP request {method!r} timed out") from exc

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    message = await self._transport.receive()
                except TransportClosed:
                    break
                self._dispatch(message)
        except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("MCP reader loop crashed")
        finally:
            # Fail any still-pending requests so callers unblock.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPClientError("transport closed before response"))
            self._pending.clear()

    def _dispatch(self, message: dict[str, Any]) -> None:
        # Responses always carry an integer id. Notifications carry no id.
        msg_id = message.get("id")
        if msg_id is None:
            # Server-initiated notification; v0.3 client ignores them but logs.
            logger.debug("MCP notification: %s", message.get("method"))
            return
        future = self._pending.pop(int(msg_id), None)
        if future is None or future.done():
            return
        if "error" in message:
            err = message["error"] or {}
            future.set_exception(
                JSONRPCError(
                    code=int(err.get("code", -32000)),
                    message=str(err.get("message", "unknown error")),
                    data=err.get("data"),
                )
            )
            return
        future.set_result(dict(message.get("result") or {}))
