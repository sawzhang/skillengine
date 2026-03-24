"""Base transport interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransportConfig:
    """Configuration for a transport connection."""

    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    session_id: str | None = None
    timeout: float = 30.0


class TransportBase(ABC):
    """Abstract base class for streaming transports."""

    def __init__(self, config: TransportConfig | None = None) -> None:
        self.config = config or TransportConfig()

    @abstractmethod
    async def stream(self, request: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream response data from the LLM provider.

        Args:
            request: The request payload to send.

        Yields:
            Raw bytes from the response stream.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""
        ...
