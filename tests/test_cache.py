"""Tests for cache retention configuration."""

import pytest

from skillengine.cache import get_cache_control_anthropic, get_cache_config_openai


class TestGetCacheControlAnthropic:
    """Tests for get_cache_control_anthropic."""

    def test_none_returns_none(self) -> None:
        """Cache retention 'none' should return None."""
        assert get_cache_control_anthropic("none") is None

    def test_short_returns_ephemeral(self) -> None:
        """Cache retention 'short' should return ephemeral without ttl."""
        result = get_cache_control_anthropic("short")
        assert result == {"type": "ephemeral"}

    def test_long_returns_ephemeral_with_ttl_for_anthropic_api(self) -> None:
        """Cache retention 'long' with default base_url should include ttl '1h'."""
        result = get_cache_control_anthropic("long")
        assert result == {"type": "ephemeral", "ttl": "1h"}

    def test_long_returns_ephemeral_with_ttl_for_explicit_anthropic_url(self) -> None:
        """Cache retention 'long' with explicit api.anthropic.com url should include ttl."""
        result = get_cache_control_anthropic("long", base_url="https://api.anthropic.com/v1")
        assert result == {"type": "ephemeral", "ttl": "1h"}

    def test_long_without_ttl_for_custom_base_url(self) -> None:
        """Cache retention 'long' with a custom base_url should not include ttl."""
        result = get_cache_control_anthropic("long", base_url="https://my-proxy.example.com")
        assert result == {"type": "ephemeral"}
        assert "ttl" not in result

    def test_short_with_base_url_returns_ephemeral(self) -> None:
        """Cache retention 'short' should return ephemeral regardless of base_url."""
        result = get_cache_control_anthropic("short", base_url="https://api.anthropic.com/v1")
        assert result == {"type": "ephemeral"}


class TestGetCacheConfigOpenai:
    """Tests for get_cache_config_openai."""

    def test_none_returns_none(self) -> None:
        """Cache retention 'none' should return None."""
        assert get_cache_config_openai("none") is None

    def test_short_with_session_id_returns_prompt_cache_key(self) -> None:
        """Cache retention 'short' with session_id should return prompt_cache_key."""
        result = get_cache_config_openai("short", session_id="sess_abc")
        assert result == {"prompt_cache_key": "sess_abc"}

    def test_long_returns_prompt_cache_retention(self) -> None:
        """Cache retention 'long' should include prompt_cache_retention '24h'."""
        result = get_cache_config_openai("long", session_id="sess_xyz")
        assert result == {"prompt_cache_key": "sess_xyz", "prompt_cache_retention": "24h"}

    def test_long_without_session_id(self) -> None:
        """Cache retention 'long' without session_id should still return retention."""
        result = get_cache_config_openai("long")
        assert result == {"prompt_cache_retention": "24h"}

    def test_short_without_session_id_returns_none(self) -> None:
        """Cache retention 'short' without session_id should return None (empty config)."""
        result = get_cache_config_openai("short")
        assert result is None

    def test_none_with_session_id_returns_none(self) -> None:
        """Cache retention 'none' should return None even with session_id."""
        result = get_cache_config_openai("none", session_id="sess_abc")
        assert result is None
