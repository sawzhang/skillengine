"""Cache retention configuration for prompt caching."""

from __future__ import annotations

from typing import Any, Literal

CacheRetention = Literal["none", "short", "long"]


def get_cache_control_anthropic(
    cache_retention: CacheRetention,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Get Anthropic cache_control settings based on retention level.

    Returns None if no caching, or a dict like {"type": "ephemeral"} or
    {"type": "ephemeral", "ttl": "1h"} for long retention on api.anthropic.com.
    """
    if cache_retention == "none":
        return None
    control: dict[str, Any] = {"type": "ephemeral"}
    if cache_retention == "long":
        # Only set TTL on the official Anthropic API
        if base_url is None or "api.anthropic.com" in (base_url or ""):
            control["ttl"] = "1h"
    return control


def get_cache_config_openai(
    cache_retention: CacheRetention,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Get OpenAI cache configuration based on retention level.

    Returns None if no caching, or a dict with prompt_cache_key and retention.
    """
    if cache_retention == "none":
        return None
    config: dict[str, Any] = {}
    if session_id:
        config["prompt_cache_key"] = session_id
    if cache_retention == "long":
        config["prompt_cache_retention"] = "24h"
    return config or None
