"""Cross-provider message transformation utilities."""

from __future__ import annotations

import hashlib
from typing import Any


def normalize_tool_call_id(original_id: str, max_length: int = 64) -> str:
    """Normalize a tool call ID to fit within provider limits.

    OpenAI generates long IDs (450+ chars); Anthropic requires max 64 chars.
    Uses SHA-256 hash prefix if ID exceeds max_length.
    """
    if len(original_id) <= max_length:
        return original_id
    hash_prefix = hashlib.sha256(original_id.encode()).hexdigest()[: max_length - 4]
    return f"tc_{hash_prefix}"


def transform_messages(
    messages: list[dict[str, Any]],
    target_provider: str = "anthropic",
    source_provider: str = "openai",
) -> list[dict[str, Any]]:
    """Transform messages for cross-provider compatibility.

    Handles:
    - Tool call ID normalization (OpenAI long -> Anthropic max 64)
    - Thinking block conversion (keep for same provider, convert to text for different)
    - Synthetic empty tool results for orphaned tool calls
    - Skip errored/aborted assistant messages
    """
    if source_provider == target_provider:
        return messages

    # Build ID mapping for tool call normalization
    id_mapping: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                original_id = tc.get("id", "")
                if original_id:
                    normalized = normalize_tool_call_id(original_id)
                    if normalized != original_id:
                        id_mapping[original_id] = normalized

    # Transform messages
    result: list[dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()

    # First pass: collect tool result IDs
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                tool_result_ids.add(tc_id)

    for msg in messages:
        transformed = dict(msg)
        role = msg.get("role", "")

        # Skip errored assistant messages (empty content + no tool calls)
        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if not content and not tool_calls:
                continue

            # Handle thinking blocks
            if "thinking" in msg:
                if source_provider != target_provider:
                    # Convert thinking to text prefix for different providers
                    thinking = msg["thinking"]
                    if thinking:
                        existing_content = transformed.get("content", "") or ""
                        transformed["content"] = f"[Thinking: {thinking}]\n\n{existing_content}"
                    transformed.pop("thinking", None)
                    transformed.pop("thought_signature", None)

            # Normalize tool call IDs
            if tool_calls:
                new_tool_calls = []
                for tc in tool_calls:
                    new_tc = dict(tc)
                    original_id = tc.get("id", "")
                    if original_id in id_mapping:
                        new_tc["id"] = id_mapping[original_id]
                    new_tc.pop("thought_signature", None)
                    new_tool_calls.append(new_tc)
                    seen_tool_call_ids.add(new_tc.get("id", original_id))
                transformed["tool_calls"] = new_tool_calls

        elif role == "tool":
            # Normalize tool_call_id reference
            tc_id = msg.get("tool_call_id", "")
            if tc_id in id_mapping:
                transformed["tool_call_id"] = id_mapping[tc_id]

        result.append(transformed)

    # Insert synthetic empty tool results for orphaned tool calls
    final: list[dict[str, Any]] = []
    for msg in result:
        final.append(msg)
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id", "")
                original_id = next((k for k, v in id_mapping.items() if v == tc_id), tc_id)
                if original_id not in tool_result_ids and tc_id not in tool_result_ids:
                    final.append(
                        {
                            "role": "tool",
                            "content": "",
                            "tool_call_id": tc_id,
                        }
                    )

    return final
