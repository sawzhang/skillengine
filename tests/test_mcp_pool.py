"""Tests for MCP-IN-3: spec parsing + connection pool + AgentRunner integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from skillengine.mcp import (
    InMemoryTransport,
    MCPConnectionPool,
    MCPServerSpec,
    coerce_spec,
    parse_mcp_uri,
)
from skillengine.mcp.transport import TransportClosed

# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_mcp_uri_stdio_shell_form() -> None:
    spec = parse_mcp_uri("mcp+stdio:npx -y @modelcontextprotocol/server-everything")
    assert spec.transport == "stdio"
    assert spec.command == "npx"
    assert spec.args == ["-y", "@modelcontextprotocol/server-everything"]
    assert spec.env == {}
    assert spec.cwd is None


def test_parse_mcp_uri_stdio_shell_form_handles_quoted_args() -> None:
    spec = parse_mcp_uri('mcp+stdio:python -c "print(1)"')
    assert spec.command == "python"
    assert spec.args == ["-c", "print(1)"]


def test_parse_mcp_uri_stdio_url_form_with_query() -> None:
    uri = "mcp+stdio:///usr/bin/echo?args=hi,there&env=DEBUG=1&cwd=/tmp&name=echo"
    spec = parse_mcp_uri(uri)
    assert spec.command == "/usr/bin/echo"
    assert spec.args == ["hi", "there"]
    assert spec.env == {"DEBUG": "1"}
    assert spec.cwd == "/tmp"
    assert spec.name == "echo"


def test_parse_mcp_uri_stdio_url_repeated_args() -> None:
    uri = "mcp+stdio://my-cmd?args=one&args=two"
    spec = parse_mcp_uri(uri)
    assert spec.args == ["one", "two"]


def test_parse_mcp_uri_command_json() -> None:
    uri = 'mcp+command:{"command": "node", "args": ["server.js"], "env": {"K": "V"}}'
    spec = parse_mcp_uri(uri)
    assert spec.command == "node"
    assert spec.args == ["server.js"]
    assert spec.env == {"K": "V"}


def test_parse_mcp_uri_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        parse_mcp_uri("http://example.com")


def test_parse_mcp_uri_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_mcp_uri("")
    with pytest.raises(ValueError):
        parse_mcp_uri("mcp+stdio:")


def test_parse_mcp_uri_rejects_malformed_env() -> None:
    with pytest.raises(ValueError):
        parse_mcp_uri("mcp+stdio://cmd?env=BADENTRY")


def test_parse_mcp_uri_rejects_malformed_json() -> None:
    with pytest.raises(ValueError):
        parse_mcp_uri("mcp+command:{not json")
    with pytest.raises(ValueError):
        parse_mcp_uri("mcp+command:[1, 2, 3]")  # not an object


def test_coerce_spec_accepts_strings_dicts_and_specs() -> None:
    s1 = coerce_spec("mcp+stdio:echo hi")
    assert s1.command == "echo" and s1.args == ["hi"]

    s2 = coerce_spec({"command": "cat", "args": ["a.txt"], "env": {"X": "1"}})
    assert s2.command == "cat" and s2.env == {"X": "1"}

    s3 = MCPServerSpec(command="ls")
    assert coerce_spec(s3) is s3

    with pytest.raises(TypeError):
        coerce_spec(42)


def test_spec_to_stdio_spec_requires_command() -> None:
    with pytest.raises(ValueError):
        MCPServerSpec(transport="stdio", command="").to_stdio_spec()


def test_spec_to_stdio_spec_rejects_non_stdio() -> None:
    with pytest.raises(ValueError):
        MCPServerSpec(transport="sse", command="x").to_stdio_spec()


# ---------------------------------------------------------------------------
# Fake server + transport factory for pool tests
# ---------------------------------------------------------------------------


class _FakeServer:
    """Tiny MCP server speaking just enough for the pool tests."""

    def __init__(self, transport, *, tool_name: str = "echo") -> None:
        self.transport = transport
        self.tool_name = tool_name
        self.calls: list[tuple[str, dict]] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

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
                        "protocolVersion": "2025-06-18",
                        "serverInfo": {"name": "fake", "version": "1"},
                        "capabilities": {"tools": {}},
                    },
                )
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                await self._reply(
                    msg_id,
                    {
                        "tools": [
                            {
                                "name": self.tool_name,
                                "description": "Echo the text",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                },
                            }
                        ]
                    },
                )
            elif method == "tools/call":
                params = msg.get("params") or {}
                self.calls.append((params.get("name"), params.get("arguments") or {}))
                await self._reply(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": str(params.get("arguments", {}))}],
                        "isError": False,
                    },
                )
            elif msg_id is not None:
                await self.transport.send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": "no"},
                    }
                )

    async def _reply(self, msg_id, result) -> None:
        if msg_id is None:
            return
        await self.transport.send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _make_factory(servers_by_spec: dict[str, _FakeServer]) -> Any:
    """Return a transport_factory that links each spec to a fresh fake server."""

    async def factory(spec: MCPServerSpec):
        client_t, server_t = InMemoryTransport.pair()
        server = _FakeServer(server_t, tool_name=f"{spec.command}-tool")
        server.start()
        servers_by_spec[spec.command] = server
        return client_t

    return factory


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_connect_single_server() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))
    try:
        client = await pool.connect(MCPServerSpec(command="alpha"))
        assert "alpha" in pool.list_connections()
        assert pool.get_client("alpha") is client
        tools = await client.list_tools()
        assert [t.name for t in tools] == ["alpha-tool"]
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_connect_multiple_servers() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))
    try:
        await pool.connect_all(
            [
                MCPServerSpec(command="alpha"),
                MCPServerSpec(command="beta"),
            ]
        )
        assert set(pool.list_connections()) == {"alpha", "beta"}
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_register_with_dispatcher_namespaces_tools() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))

    # Minimal dispatcher stub matching the duck-type the pool requires.
    registered: dict[str, tuple[Any, dict]] = {}

    class _Dispatcher:
        def register(self, name: str, handler, definition) -> None:
            registered[name] = (handler, definition)

        def unregister(self, name: str) -> bool:
            return registered.pop(name, None) is not None

    try:
        await pool.connect_all(
            [
                MCPServerSpec(command="alpha"),
                MCPServerSpec(command="beta"),
            ]
        )
        count = pool.register_with_dispatcher(_Dispatcher())
        assert count == 2
        # Tools must be namespaced.
        assert set(registered.keys()) == {"alpha__alpha-tool", "beta__beta-tool"}
        # Definitions follow OpenAI function-calling shape.
        _, defn = registered["alpha__alpha-tool"]
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "alpha__alpha-tool"
        # Handler signature: (name, args, ctx, on_output).
        handler, _ = registered["alpha__alpha-tool"]
        out = await handler("alpha__alpha-tool", {"text": "hi"}, None, None)
        assert "hi" in out
        # alpha server saw the un-prefixed remote name.
        assert servers["alpha"].calls == [("alpha-tool", {"text": "hi"})]
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_unregister_from_dispatcher() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))

    registered: dict[str, Any] = {}

    class _Dispatcher:
        def register(self, name, handler, definition) -> None:
            registered[name] = (handler, definition)

        def unregister(self, name) -> bool:
            return registered.pop(name, None) is not None

    dispatcher = _Dispatcher()
    try:
        await pool.connect(MCPServerSpec(command="alpha"))
        pool.register_with_dispatcher(dispatcher)
        assert "alpha__alpha-tool" in registered
        removed = pool.unregister_from_dispatcher(dispatcher)
        assert removed == 1
        assert registered == {}
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_duplicate_name_raises() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))
    try:
        await pool.connect(MCPServerSpec(command="alpha", name="dup"))
        with pytest.raises(ValueError):
            await pool.connect(MCPServerSpec(command="beta"), name="dup")
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_auto_names_when_command_collides() -> None:
    servers: dict[str, _FakeServer] = {}

    async def factory(spec: MCPServerSpec):
        client_t, server_t = InMemoryTransport.pair()
        server = _FakeServer(server_t, tool_name="t")
        server.start()
        servers.setdefault(f"srv{len(servers)}", server)
        return client_t

    pool = MCPConnectionPool(transport_factory=factory)
    try:
        await pool.connect(MCPServerSpec(command="same", name="x"))
        await pool.connect(MCPServerSpec(command="same"))  # would collide with "same"
        assert "x" in pool.list_connections()
        # Second connection should still register (with a generated name) since it
        # has a different logical default.
        names = pool.list_connections()
        assert len(names) == 2
    finally:
        await pool.aclose()
        for s in servers.values():
            await s.stop()


@pytest.mark.asyncio
async def test_pool_aclose_is_idempotent() -> None:
    servers: dict[str, _FakeServer] = {}
    pool = MCPConnectionPool(transport_factory=_make_factory(servers))
    await pool.connect(MCPServerSpec(command="alpha"))
    await pool.aclose()
    await pool.aclose()  # no error
    for s in servers.values():
        await s.stop()


# ---------------------------------------------------------------------------
# AgentRunner.connect_mcp_servers integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_runner_connect_mcp_servers_registers_tools(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from skillengine.adapters.registry import AdapterRegistry
    from skillengine.agent import AgentConfig, AgentRunner
    from skillengine.engine import SkillsEngine
    from skillengine.events import EventBus

    engine = MagicMock(spec=SkillsEngine)
    engine.get_snapshot.return_value = MagicMock(skills=[], prompt="", get_skill=lambda n: None)
    config = AgentConfig(
        model="m",
        base_url="http://localhost",
        api_key="key",
        max_turns=1,
        enable_tools=True,
        auto_execute=False,
    )
    runner = AgentRunner(
        engine=engine,
        config=config,
        events=EventBus(),
        adapter_registry=AdapterRegistry(),
    )

    # Inject our fake transport_factory via monkeypatching the pool class.
    from skillengine.mcp import pool as pool_module

    servers: dict[str, _FakeServer] = {}
    real_pool = pool_module.MCPConnectionPool

    def _patched(*args, **kwargs):
        kwargs.setdefault("transport_factory", _make_factory(servers))
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(pool_module, "MCPConnectionPool", _patched)

    try:
        pool = await runner.connect_mcp_servers([MCPServerSpec(command="alpha")])
        assert pool is not None
        tool_names = {n for n in runner._dispatcher._tools}
        assert "alpha__alpha-tool" in tool_names
    finally:
        await runner.disconnect_mcp_servers()
        for s in servers.values():
            await s.stop()
    # After disconnect the tool should be gone.
    assert "alpha__alpha-tool" not in runner._dispatcher._tools
    assert runner.mcp_pool is None
