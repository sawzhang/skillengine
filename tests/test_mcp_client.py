"""Tests for the MCP client.

These exercise the client against an in-process fake server connected via
:class:`InMemoryTransport`, so the suite needs no Node.js / external binaries.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skillengine.mcp import (
    MCP_PROTOCOL_VERSION,
    InMemoryTransport,
    JSONRPCError,
    MCPClient,
    MCPClientError,
    MCPToolError,
)
from skillengine.mcp.transport import StdioServerSpec, TransportClosed

# ---------------------------------------------------------------------------
# Fake server (runs on the other end of InMemoryTransport)
# ---------------------------------------------------------------------------


class FakeMCPServer:
    """Minimal fake MCP server for unit tests.

    Speaks just enough of the protocol: ``initialize`` handshake, ``tools/list``,
    ``tools/call``, plus a configurable hook for error injection.
    """

    def __init__(self, transport, tools=None) -> None:
        self.transport = transport
        self.tools = tools or [
            {
                "name": "echo",
                "description": "Echo the given text back",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "fail",
                "description": "Always returns isError=True",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.initialized = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="fake-mcp-server")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        while True:
            try:
                msg = await self.transport.receive()
            except TransportClosed:
                return
            method = msg.get("method")
            msg_id = msg.get("id")

            if method == "initialize":
                await self._reply(
                    msg_id,
                    {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "serverInfo": {"name": "fake", "version": "0.0.1"},
                        "capabilities": {"tools": {}},
                    },
                )
            elif method == "notifications/initialized":
                self.initialized = True
            elif method == "tools/list":
                await self._reply(msg_id, {"tools": self.tools})
            elif method == "tools/call":
                params = msg.get("params") or {}
                name = params.get("name")
                args = params.get("arguments") or {}
                self.call_log.append((name, args))
                if name == "echo":
                    await self._reply(
                        msg_id,
                        {
                            "content": [{"type": "text", "text": str(args.get("text", ""))}],
                            "isError": False,
                        },
                    )
                elif name == "fail":
                    await self._reply(
                        msg_id,
                        {
                            "content": [{"type": "text", "text": "boom"}],
                            "isError": True,
                        },
                    )
                else:
                    await self._error(msg_id, -32601, f"unknown tool {name}")
            elif method == "sleep_forever":
                # Deliberately never reply — used for timeout testing.
                continue
            else:
                if msg_id is not None:
                    await self._error(msg_id, -32601, f"method not found: {method}")

    async def _reply(self, msg_id, result) -> None:
        if msg_id is None:
            return
        await self.transport.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    async def _error(self, msg_id, code, message) -> None:
        if msg_id is None:
            return
        await self.transport.send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": code, "message": message},
            }
        )


@pytest.fixture
async def linked_pair():
    """Provide (client_transport, server) with the server already running."""
    client_t, server_t = InMemoryTransport.pair()
    server = FakeMCPServer(server_t)
    server.start()
    yield client_t, server
    await server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_handshake(linked_pair) -> None:
    client_t, server = linked_pair
    async with MCPClient.connect(client_t) as client:
        info = client.server_info
        assert info is not None
        assert info.server_name == "fake"
        assert info.protocol_version == MCP_PROTOCOL_VERSION
    # initialized notification must have been observed by the server.
    assert server.initialized is True


@pytest.mark.asyncio
async def test_list_tools_returns_definitions(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert names == ["echo", "fail"]
        assert tools[0].input_schema["properties"]["text"]["type"] == "string"


@pytest.mark.asyncio
async def test_list_tools_is_cached(linked_pair) -> None:
    client_t, server = linked_pair
    async with MCPClient.connect(client_t) as client:
        first = await client.list_tools()
        second = await client.list_tools()
        assert first is second  # cache returns same list object
        third = await client.list_tools(force_refresh=True)
        assert third == first
        # tools/list count: 2 (cached + force_refresh)
        list_calls = [c for c in server.call_log]
        # call_log only tracks tools/call, so just assert via behavior above.
        assert list_calls == []


@pytest.mark.asyncio
async def test_call_tool_success(linked_pair) -> None:
    client_t, server = linked_pair
    async with MCPClient.connect(client_t) as client:
        result = await client.call_tool("echo", {"text": "hello"})
        assert result.is_error is False
        assert result.text() == "hello"
        assert server.call_log == [("echo", {"text": "hello"})]


@pytest.mark.asyncio
async def test_call_tool_error_returned(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        result = await client.call_tool("fail")
        assert result.is_error is True
        assert "boom" in result.text()


@pytest.mark.asyncio
async def test_call_tool_raise_on_error(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        with pytest.raises(MCPToolError) as exc:
            await client.call_tool("fail", raise_on_error=True)
        assert exc.value.tool_name == "fail"
        assert exc.value.result.is_error is True


@pytest.mark.asyncio
async def test_unknown_tool_returns_jsonrpc_error(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        with pytest.raises(JSONRPCError) as exc:
            await client.call_tool("does-not-exist")
        assert exc.value.code == -32601


@pytest.mark.asyncio
async def test_tool_definitions_bridge_to_skillengine(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        await client.list_tools()
        defs = client.tool_definitions(prefix="srv")
        names = {d.name for d in defs}
        assert names == {"srv__echo", "srv__fail"}
        echo_def = next(d for d in defs if d.name == "srv__echo")
        text = await echo_def.handler({"text": "hi"})
        assert text == "hi"
        fail_def = next(d for d in defs if d.name == "srv__fail")
        err_text = await fail_def.handler({})
        assert err_text.startswith("[mcp-error]")


@pytest.mark.asyncio
async def test_tool_definitions_without_prefix(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        await client.list_tools()
        defs = client.tool_definitions()
        assert {d.name for d in defs} == {"echo", "fail"}


@pytest.mark.asyncio
async def test_tool_definitions_requires_list_tools_first(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t) as client:
        with pytest.raises(MCPClientError):
            client.tool_definitions()


@pytest.mark.asyncio
async def test_request_timeout(linked_pair) -> None:
    client_t, _server = linked_pair
    async with MCPClient.connect(client_t, default_timeout=0.1) as client:
        with pytest.raises(MCPClientError):
            # ``sleep_forever`` is a method the fake server intentionally never
            # answers; client must surface a timeout error.
            await client._request("sleep_forever", {}, timeout=0.1)


@pytest.mark.asyncio
async def test_close_fails_pending_requests() -> None:
    client_t, server_t = InMemoryTransport.pair()
    server = FakeMCPServer(server_t)
    server.start()
    client = MCPClient(client_t, default_timeout=2.0)
    await client.start()

    async def hanging():
        return await client._request("sleep_forever", {}, timeout=5.0)

    task = asyncio.create_task(hanging())
    await asyncio.sleep(0.05)
    await client.aclose()
    with pytest.raises(MCPClientError):
        await task
    await server.stop()


@pytest.mark.asyncio
async def test_context_blocks_render_in_text() -> None:
    from skillengine.mcp.protocol import ToolCallResult

    result = ToolCallResult.from_dict(
        {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///tmp/x"}},
                {"type": "weird"},
            ],
        }
    )
    text = result.text()
    assert "hello" in text
    assert "[image:image/png]" in text
    assert "[resource:file:///tmp/x]" in text
    assert "[weird]" in text


def test_stdio_server_spec_merges_env(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING", "1")
    spec = StdioServerSpec(command="echo", env={"EXTRA": "2"})
    merged = spec.merged_env()
    assert merged["EXISTING"] == "1"
    assert merged["EXTRA"] == "2"


@pytest.mark.asyncio
async def test_double_start_is_idempotent(linked_pair) -> None:
    client_t, _server = linked_pair
    client = MCPClient(client_t)
    info1 = await client.start()
    info2 = await client.start()
    assert info1 is info2
    await client.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent(linked_pair) -> None:
    client_t, _server = linked_pair
    client = MCPClient(client_t)
    await client.start()
    await client.aclose()
    await client.aclose()  # must not raise
