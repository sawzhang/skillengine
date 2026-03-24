"""
Event system for the agent runtime.

Provides a typed EventBus that the AgentRunner uses to emit lifecycle events.
Handlers can observe, modify, or block agent behavior by returning result objects.

Example:
    from skillengine.events import EventBus

    bus = EventBus()

    @bus.on("before_tool_call")
    async def guard(event):
        if "rm -rf" in event.args.get("command", ""):
            return ToolCallEventResult(block=True, reason="Dangerous command")
        return ToolCallEventResult()

    # In the agent loop:
    result = await bus.emit("before_tool_call", BeforeToolCallEvent(...))
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from skillengine.logging import get_logger

logger = get_logger("events")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

# Event name constants
AGENT_START = "agent_start"
AGENT_END = "agent_end"
TURN_START = "turn_start"
TURN_END = "turn_end"
BEFORE_TOOL_CALL = "before_tool_call"
AFTER_TOOL_RESULT = "after_tool_result"
CONTEXT_TRANSFORM = "context_transform"
INPUT = "input"
TOOL_EXECUTION_UPDATE = "tool_execution_update"
SESSION_START = "session_start"
SESSION_END = "session_end"
MODEL_CHANGE = "model_change"
COMPACTION = "compaction"


@dataclass
class AgentStartEvent:
    """Emitted before the first LLM call in a chat() invocation."""

    user_input: str
    system_prompt: str
    model: str
    turn: int = 0


@dataclass
class AgentEndEvent:
    """Emitted after the agent loop finishes."""

    user_input: str
    total_turns: int
    finish_reason: str = ""  # "complete", "max_turns", "error"
    error: str | None = None
    messages: list[Any] | None = None  # Conversation messages at end of loop


@dataclass
class TurnStartEvent:
    """Emitted before each LLM round-trip."""

    turn: int
    message_count: int  # number of messages being sent to LLM


@dataclass
class TurnEndEvent:
    """Emitted after each LLM round-trip."""

    turn: int
    has_tool_calls: bool
    content: str = ""
    tool_call_count: int = 0


@dataclass
class BeforeToolCallEvent:
    """Emitted before a tool is executed. Handler can block or modify args."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    turn: int


@dataclass
class ToolCallEventResult:
    """Result returned by a before_tool_call handler."""

    block: bool = False
    reason: str = ""
    modified_args: dict[str, Any] | None = None  # None = no modification


@dataclass
class AfterToolResultEvent:
    """Emitted after a tool returns its result."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    result: str
    turn: int


@dataclass
class ToolResultEventResult:
    """Result returned by an after_tool_result handler."""

    modified_result: str | None = None  # None = no modification


@dataclass
class ContextTransformEvent:
    """Emitted before messages are sent to LLM. Handler can prune/inject."""

    messages: list[Any]  # list[AgentMessage], avoid circular import
    turn: int


@dataclass
class ContextTransformEventResult:
    """Result returned by a context_transform handler."""

    messages: list[Any] | None = None  # None = no modification


@dataclass
class InputEvent:
    """Emitted when user input is received, before any processing."""

    user_input: str


@dataclass
class InputEventResult:
    """Result returned by an input handler."""

    action: str = "continue"  # "continue", "transform", "handled"
    transformed_input: str | None = None  # used when action="transform"
    response: str | None = None  # used when action="handled"


@dataclass
class ToolExecutionUpdateEvent:
    """Emitted when a tool produces intermediate output during execution."""

    tool_call_id: str
    tool_name: str
    output: str
    turn: int


@dataclass
class SessionStartEvent:
    """Emitted when a new session starts or is resumed."""

    session_id: str
    cwd: str
    resumed: bool = False


@dataclass
class SessionEndEvent:
    """Emitted when a session ends."""

    session_id: str
    entry_count: int = 0


@dataclass
class ModelChangeEvent:
    """Emitted when the model is changed mid-session."""

    previous_model: str
    new_model: str
    previous_provider: str = ""
    new_provider: str = ""


@dataclass
class CompactionEvent:
    """Emitted when context compaction occurs."""

    summary: str
    tokens_before: int = 0
    tokens_after: int = 0
    first_kept_entry_id: str | None = None


# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """
    A structured event emitted during LLM streaming or agent execution.

    Unlike plain text deltas, StreamEvents distinguish between text content,
    thinking/reasoning, tool calls, tool results, and lifecycle markers.
    This allows UIs to render each type differently.

    Event type lifecycle for a single LLM response:
        text_start → text_delta* → text_end
        thinking_start → thinking_delta* → thinking_end
        tool_call_start → tool_call_delta* → tool_call_end

    Agent-level events (only from AgentRunner.chat_stream_events):
        turn_start / turn_end
        tool_result
        done / error
    """

    type: str
    """Event type. One of:
    - ``text_start``, ``text_delta``, ``text_end``
    - ``thinking_start``, ``thinking_delta``, ``thinking_end``
    - ``tool_call_start``, ``tool_call_delta``, ``tool_call_end``
    - ``tool_result``, ``tool_output``
    - ``turn_start``, ``turn_end``
    - ``done``, ``error``
    """

    content: str = ""
    """Text content (for text_delta, thinking_delta) or result (for tool_result)."""

    tool_name: str | None = None
    """Tool name (for tool_call_start/delta/end and tool_result)."""

    tool_call_id: str | None = None
    """Tool call ID (for tool_call_start/delta/end and tool_result)."""

    turn: int = 0
    """Current turn number (for turn_start, turn_end, and tool events)."""

    error: str | None = None
    """Error message (for error events)."""

    finish_reason: str | None = None
    """Finish reason (for done events): 'complete', 'max_turns'."""

    args_delta: str | None = None
    """Partial JSON arguments (for tool_call_delta)."""

    parsed_args: dict[str, Any] | None = None
    """Parsed partial arguments (for tool_call_delta, when streaming JSON parsing is active)."""


# ---------------------------------------------------------------------------
# Handler types
# ---------------------------------------------------------------------------

# Handlers can be sync or async, and optionally return a result object.
EventHandler = Callable[..., Any]


@dataclass
class _HandlerEntry:
    """Internal: a registered handler with metadata."""

    event: str
    handler: EventHandler
    priority: int = 0  # lower runs first
    source: str = ""  # who registered it (extension name, etc.)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """
    A typed event bus for agent lifecycle events.

    Handlers are called in priority order (lower first). Handlers can be sync
    or async. Some events support handler return values that modify agent
    behavior (e.g., blocking a tool call).

    Usage:
        bus = EventBus()

        # Decorator style
        @bus.on("agent_start")
        def on_start(event: AgentStartEvent):
            print(f"Agent starting with model {event.model}")

        # Method style
        def on_end(event: AgentEndEvent):
            print(f"Agent done in {event.total_turns} turns")

        unsub = bus.on("agent_end", on_end)
        unsub()  # remove handler
    """

    def __init__(self) -> None:
        self._handlers: list[_HandlerEntry] = []

    def on(
        self,
        event: str,
        handler: EventHandler | None = None,
        priority: int = 0,
        source: str = "",
    ) -> Callable[[], None] | Callable[[EventHandler], EventHandler]:
        """
        Register an event handler.

        Can be used as a method call or as a decorator:

            # Method call — returns unsubscribe function
            unsub = bus.on("agent_start", my_handler)
            unsub()

            # Decorator — returns the original function
            @bus.on("agent_start")
            def my_handler(event):
                ...
        """
        if handler is not None:
            entry = _HandlerEntry(event=event, handler=handler, priority=priority, source=source)
            self._handlers.append(entry)

            def unsubscribe() -> None:
                try:
                    self._handlers.remove(entry)
                except ValueError:
                    pass

            return unsubscribe

        # Decorator usage: bus.on("event_name") returns a decorator
        def decorator(fn: EventHandler) -> EventHandler:
            self.on(event, fn, priority=priority, source=source)
            return fn

        return decorator

    def off(self, event: str, handler: EventHandler) -> None:
        """Remove a specific handler for an event."""
        self._handlers = [
            h for h in self._handlers if not (h.event == event and h.handler is handler)
        ]

    def off_by_source(self, source: str) -> int:
        """Remove all handlers registered by a given source. Returns count removed."""
        before = len(self._handlers)
        self._handlers = [h for h in self._handlers if h.source != source]
        return before - len(self._handlers)

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers, or all handlers for a specific event."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers = [h for h in self._handlers if h.event != event]

    async def emit(self, event: str, data: Any = None) -> list[Any]:
        """
        Emit an event and collect handler results.

        Handlers are called in priority order (lower first).
        Both sync and async handlers are supported.

        Args:
            event: Event name
            data: Event data object (e.g., AgentStartEvent)

        Returns:
            List of non-None results from handlers
        """
        relevant = sorted(
            (h for h in self._handlers if h.event == event),
            key=lambda h: h.priority,
        )

        results: list[Any] = []
        for entry in relevant:
            try:
                result = entry.handler(data)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    result = await result
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    "Event handler error (event=%s, source=%s): %s",
                    event,
                    entry.source,
                    e,
                )
        return results

    def emit_sync(self, event: str, data: Any = None) -> list[Any]:
        """
        Emit an event synchronously (only calls sync handlers).

        Async handlers are skipped with a warning.
        """
        relevant = sorted(
            (h for h in self._handlers if h.event == event),
            key=lambda h: h.priority,
        )

        results: list[Any] = []
        for entry in relevant:
            try:
                result = entry.handler(data)
                if asyncio.iscoroutine(result):
                    # Can't await in sync context — close the coroutine and warn
                    result.close()
                    logger.warning(
                        "Async handler skipped in sync emit (event=%s, source=%s)",
                        event,
                        entry.source,
                    )
                    continue
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    "Event handler error (event=%s, source=%s): %s",
                    event,
                    entry.source,
                    e,
                )
        return results

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers."""
        return len(self._handlers)

    def has_handlers(self, event: str) -> bool:
        """Check if any handlers are registered for an event."""
        return any(h.event == event for h in self._handlers)
