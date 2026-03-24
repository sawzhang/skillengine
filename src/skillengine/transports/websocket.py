"""WebSocket transport with session caching."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from skillengine.transports.base import TransportBase, TransportConfig

_SESSION_TTL = 300  # 5 minutes


class WebSocketTransport(TransportBase):
    """WebSocket transport with connection caching and automatic expiry."""

    def __init__(self, config: TransportConfig | None = None) -> None:
        super().__init__(config)
        self._connection: Any = None
        self._last_used: float = 0.0
        self._expiry_task: asyncio.Task[None] | None = None

    async def _acquire(self) -> Any:
        """Acquire (or reuse) a WebSocket connection."""
        now = time.monotonic()
        if self._connection is not None and (now - self._last_used) < _SESSION_TTL:
            self._last_used = now
            return self._connection

        # Close stale connection
        await self._release()

        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "WebSocket transport requires the 'websockets' package. "
                "Install with: pip install skillengine[websockets]"
            )

        extra_headers = self.config.headers or {}
        self._connection = await websockets.connect(
            self.config.url,
            additional_headers=extra_headers,
        )
        self._last_used = now

        # Schedule automatic expiry
        if self._expiry_task is not None:
            self._expiry_task.cancel()
        self._expiry_task = asyncio.create_task(self._expire_after(_SESSION_TTL))

        return self._connection

    async def _release(self) -> None:
        """Close the current connection if any."""
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                pass
            self._connection = None

    async def _expire_after(self, seconds: float) -> None:
        """Close the connection after *seconds* of inactivity."""
        try:
            await asyncio.sleep(seconds)
            if self._connection is not None:
                elapsed = time.monotonic() - self._last_used
                if elapsed >= seconds:
                    await self._release()
        except asyncio.CancelledError:
            pass

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream response data over a WebSocket connection."""
        import json as _json

        ws = await self._acquire()
        await ws.send(_json.dumps(request))

        async for message in ws:
            if isinstance(message, str):
                yield message.encode("utf-8")
            else:
                yield message

    async def close(self) -> None:
        """Close the transport and cancel any expiry tasks."""
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            self._expiry_task = None
        await self._release()
