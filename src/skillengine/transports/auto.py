"""Auto transport — tries WebSocket first, falls back to SSE."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from skillengine.transports.base import TransportBase, TransportConfig
from skillengine.transports.sse import SSETransport
from skillengine.transports.websocket import WebSocketTransport


class AutoTransport(TransportBase):
    """Automatically selects the best available transport.

    Tries WebSocket first; if the connection fails or the ``websockets``
    package is not installed, falls back to SSE.
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        super().__init__(config)
        self._active: TransportBase | None = None

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream using the best available transport."""
        if self._active is None:
            try:
                ws = WebSocketTransport(self.config)
                # Attempt to acquire a connection; if it fails, fall back.
                await ws._acquire()
                self._active = ws
            except Exception:
                self._active = SSETransport(self.config)

        async for chunk in self._active.stream(request):
            yield chunk

    async def close(self) -> None:
        """Close the active transport."""
        if self._active is not None:
            await self._active.close()
            self._active = None
