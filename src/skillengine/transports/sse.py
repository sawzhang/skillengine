"""SSE transport — marker class.

Actual SSE streaming is handled by the provider SDKs (openai, anthropic).
This class exists so the transport layer has a concrete SSE implementation
that can be referenced and used in auto-detection logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from skillengine.transports.base import TransportBase, TransportConfig


class SSETransport(TransportBase):
    """Server-Sent Events transport (delegated to provider SDK)."""

    def __init__(self, config: TransportConfig | None = None) -> None:
        super().__init__(config)

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[bytes]:
        """SSE streaming is handled by the provider SDK; this is a no-op."""
        return
        yield  # pragma: no cover — make this a proper async generator

    async def close(self) -> None:
        """No resources to release for the SDK-managed SSE transport."""
        pass
