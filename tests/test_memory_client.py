"""Tests for OpenVikingClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig


@pytest.fixture
def config():
    return MemoryConfig(base_url="http://localhost:1933", timeout=5.0)


@pytest.fixture
def client(config):
    return OpenVikingClient(config)


class MockResponse:
    """Minimal mock for httpx.Response."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestHealth:
    @pytest.mark.asyncio
    async def test_healthy(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=MockResponse(200))
        client._client = mock_http

        result = await client.health()
        assert result is True
        mock_http.get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_unhealthy(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=MockResponse(500))
        client._client = mock_http

        result = await client.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
        client._client = mock_http

        result = await client.health()
        assert result is False


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"session_id": "sess-123"})
        )
        client._client = mock_http

        sid = await client.create_session()
        assert sid == "sess-123"

    @pytest.mark.asyncio
    async def test_with_metadata(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"session_id": "sess-456"})
        )
        client._client = mock_http

        sid = await client.create_session(metadata={"model": "test"})
        assert sid == "sess-456"
        call_args = mock_http.post.call_args
        assert call_args[1]["json"]["metadata"] == {"model": "test"}

    @pytest.mark.asyncio
    async def test_failure_returns_none(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("timeout"))
        client._client = mock_http

        sid = await client.create_session()
        assert sid is None


class TestAddMessage:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=MockResponse(200))
        client._client = mock_http

        result = await client.add_message("sess-1", "user", "hello")
        assert result is True
        mock_http.post.assert_called_once_with(
            "/api/v1/sessions/sess-1/messages",
            json={"role": "user", "content": "hello"},
        )

    @pytest.mark.asyncio
    async def test_failure(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("error"))
        client._client = mock_http

        result = await client.add_message("sess-1", "user", "hello")
        assert result is False


class TestCommitSession:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=MockResponse(200))
        client._client = mock_http

        result = await client.commit_session("sess-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_failure(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("error"))
        client._client = mock_http

        result = await client.commit_session("sess-1")
        assert result is False


class TestFind:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"results": [{"content": "memory 1"}]})
        )
        client._client = mock_http

        results = await client.find("test query", limit=3)
        assert results == [{"content": "memory 1"}]
        call_args = mock_http.post.call_args
        assert call_args[1]["json"]["query"] == "test query"
        assert call_args[1]["json"]["limit"] == 3

    @pytest.mark.asyncio
    async def test_with_target_uri(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"results": []})
        )
        client._client = mock_http

        await client.find("q", target_uri="viking://user/memories/")
        call_args = mock_http.post.call_args
        assert call_args[1]["json"]["target_uri"] == "viking://user/memories/"

    @pytest.mark.asyncio
    async def test_failure_returns_none(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("error"))
        client._client = mock_http

        result = await client.find("q")
        assert result is None


class TestSearch:
    @pytest.mark.asyncio
    async def test_with_session(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"results": [{"content": "found"}]})
        )
        client._client = mock_http

        results = await client.search("q", session_id="sess-1")
        assert results == [{"content": "found"}]
        call_args = mock_http.post.call_args
        assert call_args[1]["json"]["session_id"] == "sess-1"


class TestLs:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            return_value=MockResponse(200, {"entries": [{"name": "prefs", "type": "directory"}]})
        )
        client._client = mock_http

        entries = await client.ls()
        assert entries == [{"name": "prefs", "type": "directory"}]

    @pytest.mark.asyncio
    async def test_recursive(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            return_value=MockResponse(200, {"entries": []})
        )
        client._client = mock_http

        await client.ls(recursive=True)
        call_args = mock_http.get.call_args
        assert call_args[1]["params"]["recursive"] == "true"

    @pytest.mark.asyncio
    async def test_failure(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("error"))
        client._client = mock_http

        result = await client.ls()
        assert result is None


class TestAddResource:
    @pytest.mark.asyncio
    async def test_success(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=MockResponse(200, {"uri": "viking://knowledge/file.py"})
        )
        client._client = mock_http

        uri = await client.add_resource("/path/to/file.py", reason="important code")
        assert uri == "viking://knowledge/file.py"

    @pytest.mark.asyncio
    async def test_failure(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("error"))
        client._client = mock_http

        uri = await client.add_resource("/path/to/file.py")
        assert uri is None


class TestInitialize:
    @pytest.mark.asyncio
    async def test_sets_available_on_healthy(self, client):
        with patch("skillengine.memory.client.OpenVikingClient.health", return_value=True):
            # Mock httpx import
            mock_httpx = MagicMock()
            mock_async_client = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_async_client
            mock_httpx.Timeout.return_value = MagicMock()

            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                result = await client.initialize()

        assert result is True
        assert client.available is True

    @pytest.mark.asyncio
    async def test_unavailable_on_unhealthy(self, client):
        with patch("skillengine.memory.client.OpenVikingClient.health", return_value=False):
            mock_httpx = MagicMock()
            mock_async_client = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_async_client
            mock_httpx.Timeout.return_value = MagicMock()

            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                result = await client.initialize()

        assert result is False
        assert client.available is False


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, client):
        mock_http = AsyncMock()
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None
