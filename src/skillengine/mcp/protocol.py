"""MCP protocol types (JSON-RPC 2.0 over MCP envelope).

Reference: https://modelcontextprotocol.io/specification

We model only the subset SkillEngine needs as a *client* in v0.3:
- ``initialize`` handshake
- ``tools/list``
- ``tools/call``

All payloads are JSON-serializable plain dicts; this module exposes typed wrappers
for ergonomics but does not enforce schema validation at parse time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MCP_PROTOCOL_VERSION = "2025-06-18"
"""MCP protocol version advertised by the SkillEngine client.

Servers may negotiate down to an older version; we accept any version the server
returns in its ``initialize`` response.
"""

JSONRPC_VERSION = "2.0"


@dataclass
class JSONRPCError(Exception):
    """Structured JSON-RPC 2.0 error."""

    code: int
    message: str
    data: Any = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"JSON-RPC error {self.code}: {self.message}"


@dataclass
class InitializeResult:
    """Server response to ``initialize`` request."""

    protocol_version: str
    server_name: str
    server_version: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    instructions: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InitializeResult:
        info = data.get("serverInfo") or {}
        return cls(
            protocol_version=str(data.get("protocolVersion") or ""),
            server_name=str(info.get("name") or ""),
            server_version=str(info.get("version") or ""),
            capabilities=dict(data.get("capabilities") or {}),
            instructions=data.get("instructions"),
            raw=dict(data),
        )


@dataclass
class MCPTool:
    """A tool advertised by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPTool:
        return cls(
            name=str(data["name"]),
            description=str(data.get("description") or ""),
            input_schema=dict(data.get("inputSchema") or {"type": "object", "properties": {}}),
            raw=dict(data),
        )


@dataclass
class ToolCallResult:
    """Result returned by ``tools/call``."""

    content: list[dict[str, Any]]
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallResult:
        return cls(
            content=list(data.get("content") or []),
            is_error=bool(data.get("isError", False)),
            raw=dict(data),
        )

    def text(self) -> str:
        """Concatenate every ``text`` content block into a single string.

        Non-text blocks (images, resource refs) are rendered as a short marker
        so they don't get silently dropped from the LLM-visible result.
        """
        parts: list[str] = []
        for block in self.content:
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "image":
                mime = block.get("mimeType", "image/*")
                parts.append(f"[image:{mime}]")
            elif btype == "resource":
                uri = (block.get("resource") or {}).get("uri", "")
                parts.append(f"[resource:{uri}]")
            else:
                parts.append(f"[{btype or 'unknown'}]")
        return "\n".join(parts)


def make_request(
    method: str,
    request_id: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request envelope."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification envelope (no id, no response expected)."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg
