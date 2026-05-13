"""MCP server specs — parsable URI/dict descriptions of remote MCP servers.

These are the **declarative** counterpart of :class:`StdioServerSpec`:
something a user can write in a config file or pass as a string. They are
resolved to live ``MCPClient`` connections by :class:`MCPConnectionPool`.

Supported string formats:

* ``mcp+stdio:<command> [args...]`` — shell-style command line
* ``mcp+stdio://<command>?args=arg1,arg2&env=K=V`` — URI with query
* ``mcp+command:<json>`` — JSON-encoded ``{"command": ..., "args": ...}``

The shell-style form is the most ergonomic and is what we recommend in docs::

    mcp+stdio:npx -y @modelcontextprotocol/server-everything
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .transport import StdioServerSpec


@dataclass
class MCPServerSpec:
    """A parsed, transport-agnostic description of an MCP server."""

    transport: str = "stdio"
    name: str | None = None
    """Optional logical name used as the prefix for the server's tools."""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    def to_stdio_spec(self) -> StdioServerSpec:
        if self.transport != "stdio":
            raise ValueError(f"Only stdio is supported in v0.3 (got {self.transport!r})")
        if not self.command:
            raise ValueError("MCPServerSpec.command is required for stdio transport")
        return StdioServerSpec(
            command=self.command,
            args=list(self.args),
            env=dict(self.env) or None,
            cwd=self.cwd,
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_mcp_uri(uri: str) -> MCPServerSpec:
    """Parse a string into an :class:`MCPServerSpec`.

    See module docstring for accepted formats.
    """
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("MCP URI must be a non-empty string")

    text = uri.strip()
    lower = text.lower()

    if lower.startswith("mcp+stdio://"):
        return _parse_stdio_url(text[len("mcp+stdio://") :])

    if lower.startswith("mcp+stdio:"):
        return _parse_stdio_shell(text[len("mcp+stdio:") :])

    if lower.startswith("mcp+command:"):
        return _parse_command_json(text[len("mcp+command:") :])

    raise ValueError(
        f"Unsupported MCP URI scheme: {uri!r}. "
        "Expected one of: mcp+stdio:..., mcp+stdio://..., mcp+command:{json}"
    )


def _parse_stdio_shell(body: str) -> MCPServerSpec:
    body = body.strip()
    if not body:
        raise ValueError("mcp+stdio: URI must include a command")
    try:
        parts = shlex.split(body)
    except ValueError as exc:
        raise ValueError(f"Invalid shell syntax in MCP URI: {exc}") from exc
    if not parts:
        raise ValueError("mcp+stdio: URI must include a command")
    return MCPServerSpec(transport="stdio", command=parts[0], args=parts[1:])


def _parse_stdio_url(body: str) -> MCPServerSpec:
    parsed = urlparse(f"//{body}")
    # ``mcp+stdio://cmd?...`` → netloc="cmd", path=""
    # ``mcp+stdio:///abs/path?...`` → netloc="", path="/abs/path"
    command = parsed.netloc or unquote(parsed.path)
    if not command:
        raise ValueError("mcp+stdio:// URI must include a command in the path")
    query = parse_qs(parsed.query, keep_blank_values=False)

    args: list[str] = []
    for raw in query.get("args", []):
        # Allow either repeated ?args=a&args=b or a single ?args=a,b,c form.
        for piece in raw.split(","):
            piece = piece.strip()
            if piece:
                args.append(piece)

    env: dict[str, str] = {}
    for raw in query.get("env", []):
        for piece in raw.split(","):
            if "=" in piece:
                k, v = piece.split("=", 1)
                env[k.strip()] = v
            elif piece.strip():
                raise ValueError(f"Malformed env entry (expected KEY=VAL): {piece!r}")

    cwd_values = query.get("cwd", [])
    cwd = cwd_values[0] if cwd_values else None

    name_values = query.get("name", [])
    name = name_values[0] if name_values else None

    return MCPServerSpec(
        transport="stdio",
        name=name,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
    )


def _parse_command_json(body: str) -> MCPServerSpec:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mcp+command: body must be valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("mcp+command: body must be a JSON object")
    return MCPServerSpec(
        transport=str(data.get("transport") or "stdio"),
        name=data.get("name"),
        command=str(data.get("command") or ""),
        args=[str(a) for a in (data.get("args") or [])],
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        cwd=data.get("cwd"),
    )


def coerce_spec(value: Any) -> MCPServerSpec:
    """Accept str / dict / :class:`MCPServerSpec`; return a normalised spec."""
    if isinstance(value, MCPServerSpec):
        return value
    if isinstance(value, str):
        return parse_mcp_uri(value)
    if isinstance(value, dict):
        return MCPServerSpec(
            transport=str(value.get("transport") or "stdio"),
            name=value.get("name"),
            command=str(value.get("command") or ""),
            args=[str(a) for a in (value.get("args") or [])],
            env={str(k): str(v) for k, v in (value.get("env") or {}).items()},
            cwd=value.get("cwd"),
        )
    raise TypeError(f"Cannot coerce {type(value).__name__} to MCPServerSpec")


__all__ = ["MCPServerSpec", "coerce_spec", "parse_mcp_uri"]
