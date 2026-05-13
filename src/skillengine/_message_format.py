"""Pure helpers for converting between :class:`AgentMessage` and adapter formats.

Extracted from :mod:`skillengine.agent` to keep the AgentRunner class focused
on orchestration. None of the functions here touch ``AgentRunner`` state —
all dependencies are passed in explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skillengine.model_registry import TokenUsage
from skillengine.models import ImageContent, TextContent

if TYPE_CHECKING:
    from skillengine.agent import AgentMessage


def format_content_for_openai(
    content: str | list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """Format message content for the OpenAI Chat Completions API."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            data_url = f"data:{block.mime_type};base64,{block.data}"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    return parts if parts else ""


def format_messages_for_openai(
    messages: list[AgentMessage],
    *,
    system_prompt: str = "",
    supports_vision: bool = False,
) -> list[dict[str, Any]]:
    """Format an AgentMessage list for the OpenAI Chat Completions API.

    Args:
        messages: Conversation messages.
        system_prompt: Optional system prompt prepended as a ``system`` role
            message when non-empty.
        supports_vision: Whether the target model accepts image inputs. When
            False, image blocks are dropped in favor of the text representation.
    """
    formatted: list[dict[str, Any]] = []

    if system_prompt:
        formatted.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "tool":
            content = msg.text_content if not isinstance(msg.content, str) else msg.content
            formatted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        elif msg.tool_calls:
            content = (
                (msg.text_content or None)
                if not isinstance(msg.content, str)
                else (msg.content or None)
            )
            formatted.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
        elif msg.has_images and supports_vision:
            formatted.append(
                {
                    "role": msg.role,
                    "content": format_content_for_openai(msg.content),
                }
            )
        else:
            content = msg.text_content if not isinstance(msg.content, str) else msg.content
            formatted.append({"role": msg.role, "content": content})

    return formatted


def convert_to_adapter_messages(messages: list[AgentMessage]) -> list[Any]:
    """Convert AgentMessages to adapter ``Message`` format."""
    from skillengine.adapters.base import Message

    result: list[Message] = []
    for msg in messages:
        metadata: dict[str, Any] = dict(msg.metadata) if msg.metadata else {}
        if msg.tool_call_id:
            metadata["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            metadata["tool_calls"] = msg.tool_calls
        if msg.name:
            metadata["name"] = msg.name
        result.append(Message(role=msg.role, content=msg.content, metadata=metadata))
    return result


def build_assistant_message_from_response(response: Any) -> tuple[AgentMessage, TokenUsage]:
    """Convert an adapter ``AgentResponse`` to an :class:`AgentMessage`.

    Returns a ``(message, token_usage)`` tuple so the caller can accumulate
    cumulative usage on the runner. Keeping the side effect at the call site
    means this helper stays pure.
    """
    # Late import to avoid a circular import at module load time.
    from skillengine.agent import AgentMessage as _AgentMessage

    token_usage = response.token_usage or TokenUsage()
    msg = _AgentMessage(
        role="assistant",
        content=response.content,
        tool_calls=response.tool_calls or [],
        token_usage=token_usage,
        metadata={
            "finish_reason": response.finish_reason or "",
            "usage": response.usage or {},
        },
    )
    return msg, token_usage


__all__ = [
    "build_assistant_message_from_response",
    "convert_to_adapter_messages",
    "format_content_for_openai",
    "format_messages_for_openai",
]
