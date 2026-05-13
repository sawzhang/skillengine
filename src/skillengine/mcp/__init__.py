"""MCP (Model Context Protocol) client integration.

This module provides a lightweight MCP client that lets SkillEngine consume tools
exposed by any MCP server (stdio transport for v0.3; SSE planned).

Typical use::

    from skillengine.mcp import MCPClient, StdioServerSpec

    async with MCPClient.connect_stdio(
        StdioServerSpec(command="npx", args=["-y", "@modelcontextprotocol/server-everything"])
    ) as client:
        for tool in client.tool_definitions(prefix="everything"):
            registry.register(tool)
"""

from __future__ import annotations

from .client import MCPClient, MCPClientError, MCPToolError
from .pool import MCPConnectionPool
from .protocol import (
    MCP_PROTOCOL_VERSION,
    InitializeResult,
    JSONRPCError,
    MCPTool,
    ToolCallResult,
)
from .server import (
    MCPServer,
    ServerTool,
    serve_stdio,
    skill_as_server_tool,
    tool_definition_as_server_tool,
)
from .spec import MCPServerSpec, coerce_spec, parse_mcp_uri
from .transport import (
    InMemoryTransport,
    StdioServerSpec,
    StdioTransport,
    Transport,
    TransportClosed,
)

__all__ = [
    "InMemoryTransport",
    "InitializeResult",
    "JSONRPCError",
    "MCPClient",
    "MCPClientError",
    "MCPConnectionPool",
    "MCPServer",
    "MCPServerSpec",
    "MCPTool",
    "MCPToolError",
    "MCP_PROTOCOL_VERSION",
    "ServerTool",
    "StdioServerSpec",
    "StdioTransport",
    "ToolCallResult",
    "Transport",
    "TransportClosed",
    "coerce_spec",
    "parse_mcp_uri",
    "serve_stdio",
    "skill_as_server_tool",
    "tool_definition_as_server_tool",
]
