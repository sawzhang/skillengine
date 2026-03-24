"""Tests for transport abstractions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.transports.base import TransportBase, TransportConfig
from skillengine.transports.sse import SSETransport
from skillengine.transports.websocket import WebSocketTransport


# ---------------------------------------------------------------------------
# SSETransport
# ---------------------------------------------------------------------------


class TestSSETransport:
    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        transport = SSETransport()
        await transport.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_stream_is_empty(self):
        transport = SSETransport()
        chunks = [c async for c in transport.stream({})]
        assert chunks == []

    def test_default_config(self):
        transport = SSETransport()
        assert transport.config.url == ""
        assert transport.config.timeout == 30.0


# ---------------------------------------------------------------------------
# WebSocketTransport
# ---------------------------------------------------------------------------


class TestWebSocketTransport:
    @pytest.mark.asyncio
    async def test_acquire_creates_connection(self):
        config = TransportConfig(url="ws://localhost:8080")
        transport = WebSocketTransport(config)

        mock_ws = AsyncMock()
        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            conn = await transport._acquire()

        assert conn is mock_ws
        assert transport._connection is mock_ws
        await transport.close()

    @pytest.mark.asyncio
    async def test_acquire_reuses_connection(self):
        config = TransportConfig(url="ws://localhost:8080")
        transport = WebSocketTransport(config)

        mock_ws = AsyncMock()
        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            conn1 = await transport._acquire()
            conn2 = await transport._acquire()

        assert conn1 is conn2
        assert mock_websockets.connect.call_count == 1
        await transport.close()

    @pytest.mark.asyncio
    async def test_release_closes_connection(self):
        config = TransportConfig(url="ws://localhost:8080")
        transport = WebSocketTransport(config)

        mock_ws = AsyncMock()
        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            await transport._acquire()

        await transport._release()
        assert transport._connection is None
        mock_ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_cancels_expiry(self):
        config = TransportConfig(url="ws://localhost:8080")
        transport = WebSocketTransport(config)

        mock_ws = AsyncMock()
        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            await transport._acquire()

        assert transport._expiry_task is not None
        await transport.close()
        assert transport._expiry_task is None
        assert transport._connection is None

    @pytest.mark.asyncio
    async def test_missing_websockets_raises(self):
        config = TransportConfig(url="ws://localhost:8080")
        transport = WebSocketTransport(config)

        with patch.dict("sys.modules", {"websockets": None}):
            with pytest.raises(ImportError, match="websockets"):
                await transport._acquire()


# ---------------------------------------------------------------------------
# AutoTransport
# ---------------------------------------------------------------------------


class TestAutoTransport:
    @pytest.mark.asyncio
    async def test_fallback_to_sse_on_failure(self):
        from skillengine.transports.auto import AutoTransport

        config = TransportConfig(url="ws://localhost:8080")
        transport = AutoTransport(config)

        with patch(
            "skillengine.transports.auto.WebSocketTransport._acquire",
            side_effect=Exception("Connection refused"),
        ):
            chunks = [c async for c in transport.stream({})]

        # Should have fallen back to SSE (which yields nothing)
        assert chunks == []
        assert isinstance(transport._active, SSETransport)
        await transport.close()

    @pytest.mark.asyncio
    async def test_uses_websocket_on_success(self):
        from skillengine.transports.auto import AutoTransport

        config = TransportConfig(url="ws://localhost:8080")
        transport = AutoTransport(config)

        mock_ws = AsyncMock()
        # Make the connection's __aiter__ return empty
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        with patch(
            "skillengine.transports.auto.WebSocketTransport._acquire",
            return_value=mock_ws,
        ):
            # Just verify the transport selected WebSocket
            # We patch _acquire on the class so it doesn't actually connect
            transport._active = None
            try:
                async for _ in transport.stream({}):
                    pass
            except Exception:
                pass  # Don't care about actual streaming, just transport selection

        assert isinstance(transport._active, WebSocketTransport)
        await transport.close()


# ---------------------------------------------------------------------------
# AgentConfig.transport default
# ---------------------------------------------------------------------------


class TestAgentConfigTransport:
    def test_default_sse(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig()
        assert config.transport == "sse"

    def test_set_websocket(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig(transport="websocket")
        assert config.transport == "websocket"

    def test_set_auto(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig(transport="auto")
        assert config.transport == "auto"
