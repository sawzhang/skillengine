"""
Transport abstractions for LLM streaming connections.

Provides SSE, WebSocket, and auto-select transport strategies.
"""

from skillengine.transports.auto import AutoTransport
from skillengine.transports.base import TransportBase, TransportConfig
from skillengine.transports.sse import SSETransport
from skillengine.transports.websocket import WebSocketTransport

__all__ = [
    "AutoTransport",
    "SSETransport",
    "TransportBase",
    "TransportConfig",
    "WebSocketTransport",
]
