"""Async HTTP client for the OpenViking Context Database REST API."""

from __future__ import annotations

from typing import Any

from skillengine.logging import get_logger
from skillengine.memory.config import MemoryConfig

logger = get_logger("memory.client")


class OpenVikingClient:
    """Thin async wrapper around the OpenViking REST API.

    All public methods return ``None`` on failure and never raise, so the
    agent can continue operating normally when the memory backend is
    unavailable.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._client: Any = None  # httpx.AsyncClient
        self.available: bool = False

    async def initialize(self) -> bool:
        """Create the HTTP client and check server health.

        Returns:
            ``True`` if the server is reachable.
        """
        try:
            import httpx
        except ImportError:
            logger.warning(
                "httpx is required for memory integration. Install with: pip install httpx"
            )
            return False

        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
        )

        self.available = await self.health()
        return self.available

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Check if the OpenViking server is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(self, metadata: dict[str, Any] | None = None) -> str | None:
        """Create a new session. Returns the session ID or ``None``."""
        try:
            body: dict[str, Any] = {}
            if metadata:
                body["metadata"] = metadata
            resp = await self._client.post("/api/v1/sessions", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("session_id") or data.get("id")
        except Exception as exc:
            logger.warning("Failed to create session: %s", exc)
            return None

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> bool:
        """Add a message to an existing session."""
        try:
            resp = await self._client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"role": role, "content": content},
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Failed to add message to session %s: %s", session_id, exc)
            return False

    async def commit_session(self, session_id: str) -> bool:
        """Trigger memory extraction for a session."""
        try:
            resp = await self._client.post(f"/api/v1/sessions/{session_id}/commit")
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Failed to commit session %s: %s", session_id, exc)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def find(
        self,
        query: str,
        target_uri: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Stateless semantic search (no session context)."""
        try:
            body: dict[str, Any] = {"query": query, "limit": limit}
            if target_uri:
                body["target_uri"] = target_uri
            resp = await self._client.post("/api/v1/search/find", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as exc:
            logger.warning("find() failed: %s", exc)
            return None

    async def search(
        self,
        query: str,
        target_uri: str | None = None,
        session_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Session-aware semantic search."""
        try:
            body: dict[str, Any] = {"query": query, "limit": limit}
            if target_uri:
                body["target_uri"] = target_uri
            if session_id:
                body["session_id"] = session_id
            resp = await self._client.post("/api/v1/search/search", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as exc:
            logger.warning("search() failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    async def ls(
        self,
        uri: str = "viking://user/memories/",
        recursive: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Browse the memory filesystem."""
        try:
            params: dict[str, Any] = {"uri": uri}
            if recursive:
                params["recursive"] = "true"
            resp = await self._client.get("/api/v1/fs/ls", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("entries", [])
        except Exception as exc:
            logger.warning("ls() failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    async def add_resource(
        self,
        path: str,
        reason: str | None = None,
    ) -> str | None:
        """Index a file or directory as a resource. Returns the URI or ``None``."""
        try:
            body: dict[str, Any] = {"path": path}
            if reason:
                body["reason"] = reason
            resp = await self._client.post("/api/v1/resources", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("uri")
        except Exception as exc:
            logger.warning("add_resource() failed: %s", exc)
            return None
