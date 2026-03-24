"""Configuration for OpenViking memory integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for connecting to an OpenViking Context Database.

    Attributes:
        base_url: OpenViking server URL.
        api_key: Optional API key for authentication.
        timeout: HTTP request timeout in seconds.
        auto_session: Automatically create an OV session on AGENT_START.
        auto_sync: Sync messages to OV on context compaction/agent end.
        auto_commit: Trigger memory extraction on AGENT_END.
        default_search_limit: Default number of results for memory search.
    """

    base_url: str = "http://localhost:1933"
    api_key: str | None = None
    timeout: float = 30.0
    auto_session: bool = True
    auto_sync: bool = True
    auto_commit: bool = True
    default_search_limit: int = 5
