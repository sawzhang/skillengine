"""Tests for the MCP server.

We exercise the server by pairing it with the existing MCP *client* through
:class:`InMemoryTransport`. This gives us end-to-end coverage of the full
JSON-RPC + handshake + tools surface in a single in-process loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from skillengine.mcp import (
    InMemoryTransport,
    MCPClient,
    MCPServer,
    ServerTool,
    ToolCallResult,
    skill_as_server_tool,
    tool_definition_as_server_tool,
)
from skillengine.models import Skill, SkillMetadata, SkillSource
from skillengine.tools.registry import ToolDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_skill(name: str = "demo", arguments_hint: str = "<query>") -> Skill:
    return Skill(
        name=name,
        description=f"Run the {name} workflow",
        content="Run with input: $ARGUMENTS",
        file_path=Path(f"/tmp/skills/{name}/SKILL.md"),
        base_dir=Path(f"/tmp/skills/{name}"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(),
        argument_hint=arguments_hint,
    )


@pytest.fixture
async def server_and_client():
    """Yield (server, client) connected via in-memory transport, both started."""
    client_t, server_t = InMemoryTransport.pair()
    server = MCPServer(server_t, name="test-server", version="9.9", instructions="hi")
    server_task = server.start()
    client = MCPClient(client_t, default_timeout=2.0)
    await client.start()
    try:
        yield server, client
    finally:
        await client.aclose()
        await server.aclose()
        # Ensure serve loop terminated.
        if not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Handshake / metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_reports_server_info(server_and_client) -> None:
    server, client = server_and_client
    info = client.server_info
    assert info is not None
    assert info.server_name == "test-server"
    assert info.server_version == "9.9"
    assert info.instructions == "hi"
    # ``initialized`` notification must have been observed by the server.
    await asyncio.wait_for(server._initialized.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_initialize_without_instructions_omits_field() -> None:
    client_t, server_t = InMemoryTransport.pair()
    server = MCPServer(server_t, name="x", version="1")
    server.start()
    try:
        async with MCPClient.connect(client_t) as client:
            assert client.server_info is not None
            assert client.server_info.instructions is None
    finally:
        await server.aclose()


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_empty(server_and_client) -> None:
    _server, client = server_and_client
    tools = await client.list_tools()
    assert tools == []


@pytest.mark.asyncio
async def test_register_skill_exposes_it(server_and_client) -> None:
    server, client = server_and_client
    server.register_skill(_make_skill("demo", arguments_hint="<query>"))
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["skill__demo"]
    schema = tools[0].input_schema
    assert schema["properties"]["arguments"]["description"] == "<query>"


@pytest.mark.asyncio
async def test_call_skill_substitutes_arguments(server_and_client) -> None:
    server, client = server_and_client
    server.register_skill(_make_skill("demo"))
    await client.list_tools()
    result = await client.call_tool("skill__demo", {"arguments": "report.pdf"})
    assert result.is_error is False
    assert result.text() == "Run with input: report.pdf"


@pytest.mark.asyncio
async def test_call_skill_without_arguments_uses_empty_string(server_and_client) -> None:
    server, client = server_and_client
    server.register_skill(_make_skill("demo"))
    await client.list_tools()
    result = await client.call_tool("skill__demo", {})
    assert result.text() == "Run with input: "


# ---------------------------------------------------------------------------
# Custom ServerTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_server_tool_str_handler(server_and_client) -> None:
    server, client = server_and_client

    async def handler(args: dict[str, Any]) -> str:
        return f"echo:{args.get('text', '')}"

    server.register_tool(
        ServerTool(
            name="echo",
            description="Echo back",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=handler,
        )
    )
    await client.list_tools()
    result = await client.call_tool("echo", {"text": "hi"})
    assert result.text() == "echo:hi"


@pytest.mark.asyncio
async def test_handler_returning_tool_call_result(server_and_client) -> None:
    server, client = server_and_client

    async def handler(_args: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            content=[{"type": "text", "text": "ok"}, {"type": "text", "text": "two"}],
            is_error=False,
        )

    server.register_tool(
        ServerTool(name="multi", description="d", input_schema={"type": "object"}, handler=handler)
    )
    await client.list_tools()
    result = await client.call_tool("multi", {})
    assert result.text() == "ok\ntwo"


@pytest.mark.asyncio
async def test_handler_returning_raw_dict(server_and_client) -> None:
    server, client = server_and_client

    async def handler(_args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": "raw"}],
            "isError": True,
        }

    server.register_tool(
        ServerTool(name="raw", description="d", input_schema={"type": "object"}, handler=handler)
    )
    await client.list_tools()
    result = await client.call_tool("raw", {})
    assert result.is_error is True
    assert result.text() == "raw"


@pytest.mark.asyncio
async def test_handler_exception_becomes_is_error(server_and_client) -> None:
    server, client = server_and_client

    async def handler(_args: dict[str, Any]) -> str:
        raise ValueError("nope")

    server.register_tool(
        ServerTool(name="boom", description="d", input_schema={"type": "object"}, handler=handler)
    )
    await client.list_tools()
    result = await client.call_tool("boom", {})
    assert result.is_error is True
    assert "ValueError" in result.text()
    assert "nope" in result.text()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_returns_method_not_found(server_and_client) -> None:
    _server, client = server_and_client
    from skillengine.mcp import JSONRPCError

    with pytest.raises(JSONRPCError) as exc:
        await client.call_tool("does-not-exist")
    assert exc.value.code == -32601


@pytest.mark.asyncio
async def test_invalid_arguments_type_returns_invalid_params(server_and_client) -> None:
    server, client = server_and_client
    server.register_skill(_make_skill("demo"))
    await client.list_tools()
    from skillengine.mcp import JSONRPCError

    # Hand-craft a malformed request: arguments must be an object.
    with pytest.raises(JSONRPCError) as exc:
        await client._request("tools/call", {"name": "skill__demo", "arguments": "wrong"})
    assert exc.value.code == -32602


@pytest.mark.asyncio
async def test_unknown_method_returns_method_not_found(server_and_client) -> None:
    _server, client = server_and_client
    from skillengine.mcp import JSONRPCError

    with pytest.raises(JSONRPCError) as exc:
        await client._request("does/not/exist", {})
    assert exc.value.code == -32601


@pytest.mark.asyncio
async def test_ping_returns_empty_result(server_and_client) -> None:
    _server, client = server_and_client
    result = await client._request("ping", {})
    assert result == {}


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_engine_skills(server_and_client, tmp_path) -> None:
    """End-to-end: build a real SkillsEngine from a tmpdir, expose via MCP."""
    server, client = server_and_client
    from skillengine import SkillsConfig, SkillsEngine

    skill_dir = tmp_path / "skills" / "hello"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        '---\nname: hello\ndescription: "Say hello"\n---\nHello $ARGUMENTS!\n'
    )

    engine = SkillsEngine(SkillsConfig(skill_dirs=[tmp_path / "skills"]))
    added = server.register_engine_skills(engine)
    assert added >= 1

    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert "skill__hello" in names

    result = await client.call_tool("skill__hello", {"arguments": "world"})
    assert "Hello world!" in result.text()


# ---------------------------------------------------------------------------
# Tool-definition bridge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_definition_as_server_tool_roundtrips(server_and_client) -> None:
    server, client = server_and_client

    async def handler(args: dict[str, Any]) -> str:
        return f"got={args}"

    td = ToolDefinition(
        name="td_demo",
        description="demo",
        parameters={"type": "object", "properties": {"x": {"type": "number"}}},
        handler=handler,
    )
    server.register_tool_definition(td)
    await client.list_tools()
    result = await client.call_tool("td_demo", {"x": 1})
    assert "got=" in result.text()
    assert "1" in result.text()


def test_tool_definition_without_handler_raises() -> None:
    td = ToolDefinition(name="x", description="d", parameters={}, handler=None)
    server_tool = tool_definition_as_server_tool(td)
    import asyncio

    with pytest.raises(RuntimeError):
        asyncio.get_event_loop().run_until_complete(server_tool.handler({}))


def test_skill_as_server_tool_uses_default_arg_description() -> None:
    skill = Skill(
        name="x",
        description="d",
        content="hi $ARGUMENTS",
        file_path=Path("/tmp/x/SKILL.md"),
        base_dir=Path("/tmp/x"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(),
    )
    st = skill_as_server_tool(skill)
    assert st.input_schema["properties"]["arguments"]["description"].startswith("Arguments")
