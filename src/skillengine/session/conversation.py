"""Conversion utilities between AgentMessage and SessionMessageEntry.

Provides bidirectional conversion so session entries can be reconstructed
into AgentMessage objects (for wake/restore) and AgentMessages can be
persisted to the session (for dual-write).
"""

from __future__ import annotations

from typing import Any


def session_entry_to_agent_message(entry: Any) -> Any:
    """Convert a SessionMessageEntry to an AgentMessage.

    Args:
        entry: A SessionMessageEntry from the session.

    Returns:
        An AgentMessage with equivalent content.

    Note:
        Uses Any types to avoid circular imports between session
        and agent modules. The actual types are SessionMessageEntry
        and AgentMessage.
    """
    # Import here to avoid circular dependency
    from skillengine.agent import AgentMessage

    return AgentMessage(
        role=entry.role,
        content=entry.content,
        tool_calls=entry.tool_calls if entry.tool_calls else [],
        tool_call_id=entry.tool_call_id,
        name=entry.name,
        metadata=entry.metadata if entry.metadata else {},
    )


def agent_message_to_session_kwargs(msg: Any) -> dict[str, Any]:
    """Convert an AgentMessage to kwargs for SessionManager.append_message().

    Args:
        msg: An AgentMessage from the agent conversation.

    Returns:
        Dict of kwargs suitable for SessionManager.append_message().
    """
    # Extract text content from multi-modal messages
    if isinstance(msg.content, str):
        content = msg.content
    else:
        # Multi-modal: extract text blocks
        parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        content = "".join(parts)

    kwargs: dict[str, Any] = {
        "role": msg.role,
        "content": content,
    }

    if msg.tool_calls:
        kwargs["tool_calls"] = msg.tool_calls
    if msg.tool_call_id is not None:
        kwargs["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        kwargs["name"] = msg.name
    if msg.metadata:
        kwargs["metadata"] = msg.metadata

    return kwargs
