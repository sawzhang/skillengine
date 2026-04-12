"""Harness state — all mutable state for an agent run.

Consolidates AgentRunner's 9 mutable instance variables into a single
object that can be serialized to a session CustomEntry and restored
from it. Aligns with Managed Agents' stateless harness pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from skillengine.model_registry import TokenUsage


@dataclass
class HarnessState:
    """All mutable state for an agent run.

    This bridges the Brain (harness logic) and Session (persistent storage).
    Persistent fields can be serialized; transient fields (signals) are
    excluded from serialization.
    """

    # --- Persistent state (survives checkpoint/restore) ---

    #: Conversation history
    conversation: list[Any] = field(default_factory=list)  # list[AgentMessage]

    #: Current model name
    active_model: str = ""

    #: Currently active adapter name
    active_adapter_name: str | None = None

    #: Thinking level
    thinking_level: str = "off"

    #: Cumulative token usage
    cumulative_usage: TokenUsage = field(default_factory=TokenUsage)

    #: Loaded context file objects
    context_files: list[Any] = field(default_factory=list)

    #: Skill change callbacks
    on_skill_change: list[Callable] = field(default_factory=list, repr=False)

    # --- Transient state (not serialized, recreated on restore) ---

    #: Abort signal
    abort_event: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False
    )

    #: Steering message queue
    steering_queue: asyncio.Queue = field(
        default_factory=asyncio.Queue, repr=False
    )

    #: Follow-up message queue
    followup_queue: asyncio.Queue = field(
        default_factory=asyncio.Queue, repr=False
    )

    #: OpenAI client (lazy-init)
    client: Any = field(default=None, repr=False)

    def to_session_entry(self) -> dict[str, Any]:
        """Serialize persistent state for session storage.

        Returns a dict suitable for CustomEntry.data.
        """
        return {
            "active_model": self.active_model,
            "active_adapter_name": self.active_adapter_name,
            "thinking_level": self.thinking_level,
            "cumulative_usage": {
                "input_tokens": self.cumulative_usage.input_tokens,
                "output_tokens": self.cumulative_usage.output_tokens,
                "cache_read_tokens": self.cumulative_usage.cache_read_tokens,
                "cache_write_tokens": self.cumulative_usage.cache_write_tokens,
            },
        }

    @classmethod
    def from_session_data(
        cls,
        data: dict[str, Any],
        conversation: list[Any] | None = None,
        context_files: list[Any] | None = None,
    ) -> HarnessState:
        """Restore state from a serialized session entry.

        Args:
            data: Dict from CustomEntry.data (as written by to_session_entry).
            conversation: Pre-built conversation list (from session messages).
            context_files: Pre-loaded context files.
        """
        usage_data = data.get("cumulative_usage", {})
        return cls(
            conversation=conversation or [],
            active_model=data.get("active_model", ""),
            active_adapter_name=data.get("active_adapter_name"),
            thinking_level=data.get("thinking_level", "off"),
            cumulative_usage=TokenUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=usage_data.get("cache_write_tokens", 0),
            ),
            context_files=context_files or [],
        )
