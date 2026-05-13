"""Transports for MCP — stdio (subprocess) and in-memory (for tests).

Each transport exposes a uniform async interface:

- ``send(message)``: serialize and write one JSON-RPC message
- ``receive()``: read and parse the next incoming JSON-RPC message
- ``close()``: shut down the underlying channel
- ``closed``: ``True`` once the peer has gone away

The MCP stdio framing is line-delimited JSON: each message occupies exactly one
line on stdout (and one on stdin), terminated by ``\\n``. This matches the
upstream reference servers (``@modelcontextprotocol/server-*``).
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class TransportClosed(RuntimeError):  # noqa: N818 - public name, not an "Error"-suffixed class
    """Raised when reading/writing a transport whose peer has gone away."""


class Transport(ABC):
    """Abstract bidirectional JSON-message transport."""

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    async def receive(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def closed(self) -> bool: ...


# ---------------------------------------------------------------------------
# Stdio transport (spawned subprocess)
# ---------------------------------------------------------------------------


@dataclass
class StdioServerSpec:
    """Specification for an MCP server launched as a subprocess."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    def merged_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        return env


class StdioTransport(Transport):
    """JSON-line transport over a child process's stdin/stdout.

    Stderr is intentionally **not** piped: MCP servers commonly emit
    human-readable diagnostics on stderr that would otherwise fill an unread
    buffer and block the subprocess. We inherit the parent's stderr so logs
    flow through naturally.
    """

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._closed = False
        self._send_lock = asyncio.Lock()

    @classmethod
    async def spawn(cls, spec: StdioServerSpec) -> StdioTransport:
        process = await asyncio.create_subprocess_exec(
            spec.command,
            *spec.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            env=spec.merged_env(),
            cwd=spec.cwd,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("subprocess pipes were not opened")
        return cls(process)

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed or self._process.stdin is None or self._process.stdin.is_closing():
            raise TransportClosed("stdio transport is closed")
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._send_lock:
            self._process.stdin.write(payload)
            await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        if self._closed or self._process.stdout is None:
            raise TransportClosed("stdio transport is closed")
        line = await self._process.stdout.readline()
        if not line:
            self._closed = True
            raise TransportClosed("server closed stdout")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportClosed(f"server emitted non-JSON line: {line!r}") from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin and not self._process.stdin.is_closing():
            try:
                self._process.stdin.close()
            except Exception:
                pass
        if self._process.returncode is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

    @property
    def closed(self) -> bool:
        return self._closed or self._process.returncode is not None


# ---------------------------------------------------------------------------
# In-memory transport (tests, in-process MCP servers)
# ---------------------------------------------------------------------------


class InMemoryTransport(Transport):
    """A pair of asyncio queues simulating an MCP peer.

    Use :meth:`pair` to get two linked transports — one for the client, one
    for the server — that route messages between each other. This is the
    backbone of our MCP client unit tests.
    """

    def __init__(
        self,
        inbound: asyncio.Queue[dict[str, Any]],
        outbound: asyncio.Queue[dict[str, Any]],
    ) -> None:
        self._inbound = inbound
        self._outbound = outbound
        self._closed = False

    @classmethod
    def pair(cls) -> tuple[InMemoryTransport, InMemoryTransport]:
        a_to_b: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        b_to_a: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        client = cls(inbound=b_to_a, outbound=a_to_b)
        server = cls(inbound=a_to_b, outbound=b_to_a)
        return client, server

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise TransportClosed("in-memory transport closed")
        await self._outbound.put(message)

    async def receive(self) -> dict[str, Any]:
        if self._closed and self._inbound.empty():
            raise TransportClosed("in-memory transport closed")
        message = await self._inbound.get()
        if message is _SENTINEL_CLOSE:
            self._closed = True
            raise TransportClosed("peer closed")
        return message

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Signal peer that we've gone away.
        await self._outbound.put(_SENTINEL_CLOSE)

    @property
    def closed(self) -> bool:
        return self._closed


_SENTINEL_CLOSE: Any = object()
