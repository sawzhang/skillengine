"""Tests for the streaming partial JSON parser."""

from __future__ import annotations

import importlib

import pytest

from skillengine.utils.json_parse import parse_streaming_json

# Detect whether partial_json_parser is available so we can conditionally test.
_has_partial_json_parser = importlib.util.find_spec("partial_json_parser") is not None


class TestParseStreamingJson:
    """Tests for parse_streaming_json."""

    # -- empty / whitespace inputs ------------------------------------------

    def test_empty_string_returns_empty_dict(self) -> None:
        """Empty string should return {}."""
        assert parse_streaming_json("") == {}

    def test_whitespace_returns_empty_dict(self) -> None:
        """Whitespace-only string should return {}."""
        assert parse_streaming_json("   ") == {}
        assert parse_streaming_json("\n\t") == {}

    # -- valid complete JSON ------------------------------------------------

    def test_valid_complete_json(self) -> None:
        """Valid complete JSON dict should be returned as-is."""
        result = parse_streaming_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_with_nested_objects(self) -> None:
        """Nested JSON objects should parse correctly."""
        result = parse_streaming_json('{"a": {"b": 1}, "c": [1, 2]}')
        assert result == {"a": {"b": 1}, "c": [1, 2]}

    def test_valid_json_with_various_types(self) -> None:
        """JSON with numbers, booleans, and null should parse correctly."""
        result = parse_streaming_json(
            '{"num": 42, "flag": true, "nothing": null}'
        )
        assert result == {"num": 42, "flag": True, "nothing": None}

    def test_valid_empty_dict(self) -> None:
        """An empty JSON object '{}' should return {}."""
        assert parse_streaming_json("{}") == {}

    # -- non-dict JSON returns {} -------------------------------------------

    def test_json_array_returns_empty_dict(self) -> None:
        """A valid JSON array (non-dict) should return {}."""
        assert parse_streaming_json("[1, 2, 3]") == {}

    def test_json_string_returns_empty_dict(self) -> None:
        """A valid JSON string (non-dict) should return {}."""
        assert parse_streaming_json('"just a string"') == {}

    def test_json_number_returns_empty_dict(self) -> None:
        """A valid JSON number (non-dict) should return {}."""
        assert parse_streaming_json("42") == {}

    def test_json_boolean_returns_empty_dict(self) -> None:
        """A valid JSON boolean (non-dict) should return {}."""
        assert parse_streaming_json("true") == {}

    def test_json_null_returns_empty_dict(self) -> None:
        """A valid JSON null (non-dict) should return {}."""
        assert parse_streaming_json("null") == {}

    # -- invalid / partial JSON ---------------------------------------------

    def test_completely_invalid_json_returns_empty_dict(self) -> None:
        """Total garbage input should return {}."""
        assert parse_streaming_json("not json at all") == {}

    def test_partial_json_parser_truncated_value(self) -> None:
        """Truncated JSON should either parse via partial_json_parser or fall back to {}."""
        result = parse_streaming_json('{"key": "val')
        if _has_partial_json_parser:
            # partial_json_parser should recover the truncated string
            assert isinstance(result, dict)
            assert "key" in result
        else:
            assert result == {}

    def test_partial_json_parser_truncated_object(self) -> None:
        """An unclosed JSON object should either parse partially or fall back."""
        result = parse_streaming_json('{"a": 1, "b": 2')
        if _has_partial_json_parser:
            assert isinstance(result, dict)
            assert result.get("a") == 1
            assert result.get("b") == 2
        else:
            assert result == {}

    def test_partial_json_parser_truncated_nested(self) -> None:
        """Truncated nested JSON should either parse partially or fall back."""
        result = parse_streaming_json('{"outer": {"inner": "va')
        if _has_partial_json_parser:
            assert isinstance(result, dict)
            assert "outer" in result
        else:
            assert result == {}

    @pytest.mark.skipif(
        not _has_partial_json_parser,
        reason="partial_json_parser library not installed",
    )
    def test_partial_json_parser_library_returns_dict(self) -> None:
        """When partial_json_parser is installed, incomplete dicts should parse."""
        result = parse_streaming_json('{"tool": "search", "query": "hel')
        assert isinstance(result, dict)
        assert result["tool"] == "search"
        assert "query" in result

    # -- edge cases ---------------------------------------------------------

    def test_json_with_unicode(self) -> None:
        """JSON with unicode characters should parse correctly."""
        result = parse_streaming_json('{"emoji": "\\u2764", "text": "ok"}')
        assert result == {"emoji": "\u2764", "text": "ok"}

    def test_json_with_escaped_quotes(self) -> None:
        """JSON with escaped quotes should parse correctly."""
        result = parse_streaming_json('{"say": "he said \\"hi\\""}')
        assert result == {"say": 'he said "hi"'}
