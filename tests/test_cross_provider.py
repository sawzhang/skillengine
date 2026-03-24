"""Tests for cross-provider message transformation utilities."""

import hashlib

import pytest

from skillengine.adapters.transform import normalize_tool_call_id, transform_messages


class TestNormalizeToolCallId:
    """Tests for normalize_tool_call_id."""

    def test_short_id_unchanged(self) -> None:
        """Short IDs should be returned as-is."""
        short_id = "call_abc123"
        assert normalize_tool_call_id(short_id) == short_id

    def test_long_id_truncated_with_hash_prefix(self) -> None:
        """Long IDs exceeding max_length should be hashed with tc_ prefix."""
        long_id = "x" * 100
        result = normalize_tool_call_id(long_id, max_length=64)

        assert result.startswith("tc_")
        assert len(result) <= 64
        # The hash portion should be deterministic
        expected_hash = hashlib.sha256(long_id.encode()).hexdigest()[:60]
        assert result == f"tc_{expected_hash}"

    def test_exact_boundary_unchanged(self) -> None:
        """An ID exactly at max_length should be returned unchanged."""
        exact_id = "a" * 64
        assert normalize_tool_call_id(exact_id, max_length=64) == exact_id

    def test_one_over_boundary_is_hashed(self) -> None:
        """An ID one character over max_length should be hashed."""
        over_id = "a" * 65
        result = normalize_tool_call_id(over_id, max_length=64)

        assert result.startswith("tc_")
        # tc_ prefix (3 chars) + hash[:max_length - 4] (60 chars) = 63 chars
        assert len(result) <= 64


class TestTransformMessages:
    """Tests for transform_messages."""

    def test_same_provider_returns_unchanged(self) -> None:
        """When source and target providers are the same, messages are returned as-is."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = transform_messages(messages, target_provider="openai", source_provider="openai")
        assert result is messages

    def test_tool_call_id_normalization(self) -> None:
        """Long tool call IDs should be normalized in both assistant and tool messages."""
        long_id = "call_" + "z" * 200
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [{"id": long_id, "function": {"name": "test"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": long_id},
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        normalized_id = normalize_tool_call_id(long_id)
        # Assistant tool call ID should be normalized
        assert result[1]["tool_calls"][0]["id"] == normalized_id
        # Tool result should reference the normalized ID
        assert result[2]["tool_call_id"] == normalized_id

    def test_thinking_block_converted_to_text_prefix(self) -> None:
        """Thinking blocks should be converted to a text prefix for cross-provider messages."""
        messages = [
            {"role": "user", "content": "think about this"},
            {
                "role": "assistant",
                "content": "my answer",
                "thinking": "let me reason about this",
                "thought_signature": "sig123",
            },
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        assistant_msg = result[1]
        assert assistant_msg["content"] == "[Thinking: let me reason about this]\n\nmy answer"
        assert "thinking" not in assistant_msg
        assert "thought_signature" not in assistant_msg

    def test_thinking_block_empty_is_not_prefixed(self) -> None:
        """Empty thinking blocks should not add a prefix."""
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer",
                "thinking": "",
            },
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        assistant_msg = result[1]
        assert assistant_msg["content"] == "answer"
        assert "thinking" not in assistant_msg

    def test_skip_empty_assistant_messages(self) -> None:
        """Assistant messages with no content and no tool calls should be skipped."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "user", "content": "try again"},
            {"role": "assistant", "content": "ok"},
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        roles = [m["role"] for m in result]
        assert roles == ["user", "user", "assistant"]
        assert result[2]["content"] == "ok"

    def test_synthetic_tool_results_for_orphaned_calls(self) -> None:
        """Orphaned tool calls (no matching tool result) should get synthetic empty results."""
        messages = [
            {"role": "user", "content": "do things"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "tool_a"}},
                    {"id": "call_2", "function": {"name": "tool_b"}},
                ],
            },
            {"role": "tool", "content": "result_a", "tool_call_id": "call_1"},
            # No tool result for call_2 - it's orphaned
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        # Should have: user, assistant, synthetic for call_2, tool for call_1
        tool_messages = [m for m in result if m["role"] == "tool"]
        tool_call_ids = {m["tool_call_id"] for m in tool_messages}
        assert "call_1" in tool_call_ids
        assert "call_2" in tool_call_ids

        # The synthetic one should have empty content
        synthetic = next(m for m in tool_messages if m["tool_call_id"] == "call_2")
        assert synthetic["content"] == ""

    def test_thought_signature_removed_from_tool_calls(self) -> None:
        """thought_signature should be stripped from individual tool calls."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "test"}, "thought_signature": "sig"},
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": "call_1"},
        ]

        result = transform_messages(messages, target_provider="anthropic", source_provider="openai")

        assert "thought_signature" not in result[1]["tool_calls"][0]
