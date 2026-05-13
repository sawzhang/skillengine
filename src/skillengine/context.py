"""
Context management pipeline.

Provides token estimation, context window tracking, and compaction strategies
to keep conversations within model limits.

Example:
    from skillengine.context import ContextManager, TokenBudgetCompactor

    ctx_mgr = ContextManager(
        context_window=128_000,
        reserve_tokens=4096,
        compactor=TokenBudgetCompactor(),
    )

    # In the agent loop:
    if ctx_mgr.should_compact(messages):
        messages = await ctx_mgr.compact(messages)
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from skillengine.models import ImageContent, TextContent

if TYPE_CHECKING:
    from skillengine.agent import AgentMessage

# Summarizer callback: takes a list of messages, returns a string (sync or async).
Summarizer = Callable[[list["AgentMessage"]], Any]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using chars/4 heuristic.

    This is a conservative overestimate suitable for budget checks.
    For accurate counts, use a tokenizer (tiktoken, etc.).
    """
    return max(1, len(text) // 4)


def estimate_content_tokens(content: Any) -> int:
    """
    Estimate token count for message content.

    Supports plain string content and multi-modal block lists
    (TextContent/ImageContent). Unknown block types fall back to a
    conservative ``str()`` estimate.
    """
    if isinstance(content, str):
        return estimate_tokens(content)

    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, TextContent):
                total += estimate_tokens(block.text)
                continue

            if isinstance(block, ImageContent):
                # Image payloads can be large; use a minimum floor so we do
                # not under-estimate context usage for vision prompts.
                payload_tokens = max(128, estimate_tokens(block.data))
                total += payload_tokens + estimate_tokens(block.mime_type) + 8
                continue

            if isinstance(block, dict):
                block_type = str(block.get("type", ""))
                if block_type == "text":
                    total += estimate_tokens(str(block.get("text", "")))
                elif block_type == "image":
                    payload_tokens = max(128, estimate_tokens(str(block.get("data", ""))))
                    total += payload_tokens + estimate_tokens(str(block.get("mime_type", ""))) + 8
                else:
                    total += estimate_tokens(str(block))
                continue

            total += estimate_tokens(str(block))

        return max(1, total)

    return estimate_tokens(str(content))


def estimate_message_tokens(message: AgentMessage) -> int:
    """
    Estimate token count for a single AgentMessage.

    Accounts for role overhead, content, tool call arguments, and reasoning.
    """
    # Base overhead per message (role, separators)
    tokens = 4

    tokens += estimate_content_tokens(message.content)

    if message.reasoning:
        tokens += estimate_tokens(message.reasoning)

    for tc in message.tool_calls:
        # Tool call overhead
        tokens += 4
        tokens += estimate_tokens(tc.get("name", ""))
        args = tc.get("arguments", "")
        if isinstance(args, dict):
            import json

            args = json.dumps(args)
        tokens += estimate_tokens(args)

    return tokens


def estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Compaction strategies
# ---------------------------------------------------------------------------


class ContextCompactor(ABC):
    """Base class for context compaction strategies."""

    @abstractmethod
    async def compact(
        self,
        messages: list[AgentMessage],
        budget_tokens: int,
    ) -> list[AgentMessage]:
        """
        Compact messages to fit within the token budget.

        Args:
            messages: Full conversation history.
            budget_tokens: Target maximum token count.

        Returns:
            Compacted message list.
        """
        ...


class SlidingWindowCompactor(ContextCompactor):
    """
    Keep the most recent N turns, discard older ones.

    A "turn" is a user message plus all subsequent messages until the next
    user message (assistant replies, tool calls/results).
    """

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns

    async def compact(
        self,
        messages: list[AgentMessage],
        budget_tokens: int,
    ) -> list[AgentMessage]:
        if not messages:
            return messages

        # Split into turns (each starts with a user message)
        turns: list[list[AgentMessage]] = []
        current_turn: list[AgentMessage] = []

        for msg in messages:
            if msg.role == "user" and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        # Keep the last max_turns turns
        kept_turns = turns[-self.max_turns :]

        result: list[AgentMessage] = []
        for turn in kept_turns:
            result.extend(turn)

        # Also check token budget — if still over, drop from the front
        while len(result) > 1 and estimate_messages_tokens(result) > budget_tokens:
            # Always keep at least the last user message
            if result[0].role == "user":
                # Drop this user turn and its associated messages
                next_user = 1
                while next_user < len(result) and result[next_user].role != "user":
                    next_user += 1
                result = result[next_user:]
            else:
                result = result[1:]

        return result


class TokenBudgetCompactor(ContextCompactor):
    """
    Remove oldest messages to fit within a token budget.

    Preserves the most recent messages while respecting tool call / tool
    result pairing (never orphan a tool result without its call).
    """

    async def compact(
        self,
        messages: list[AgentMessage],
        budget_tokens: int,
    ) -> list[AgentMessage]:
        if not messages:
            return messages

        if estimate_messages_tokens(messages) <= budget_tokens:
            return messages

        # Work backwards, keeping messages until budget is exhausted
        kept: list[AgentMessage] = []
        running_tokens = 0

        for msg in reversed(messages):
            msg_tokens = estimate_message_tokens(msg)
            if running_tokens + msg_tokens > budget_tokens and kept:
                break
            kept.append(msg)
            running_tokens += msg_tokens

        kept.reverse()

        # Ensure the first message is a user message (LLM APIs require this)
        while kept and kept[0].role not in ("user", "system"):
            kept = kept[1:]

        return kept


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """
    Manages context window budget for an agent.

    Tracks cumulative token usage, detects when compaction is needed, and
    applies a compaction strategy.

    Example:
        ctx_mgr = ContextManager(
            context_window=128_000,
            reserve_tokens=8192,
        )

        # Before each LLM call:
        if ctx_mgr.should_compact(messages):
            messages = await ctx_mgr.compact(messages)
    """

    def __init__(
        self,
        context_window: int = 128_000,
        reserve_tokens: int = 4096,
        compactor: ContextCompactor | None = None,
        threshold: float = 0.9,
    ) -> None:
        """
        Args:
            context_window: Maximum context tokens the model accepts.
            reserve_tokens: Tokens to reserve for the model's output.
            compactor: Strategy for compacting messages. Defaults to TokenBudgetCompactor.
            threshold: Trigger compaction when usage exceeds this fraction of budget.
        """
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.compactor = compactor or TokenBudgetCompactor()
        self.threshold = threshold

    @property
    def budget_tokens(self) -> int:
        """Maximum tokens available for input (context_window - reserve)."""
        return self.context_window - self.reserve_tokens

    def estimate_tokens(self, messages: list[AgentMessage]) -> int:
        """Estimate total tokens for a message list."""
        return estimate_messages_tokens(messages)

    def should_compact(self, messages: list[AgentMessage]) -> bool:
        """Check if the messages exceed the compaction threshold."""
        current = self.estimate_tokens(messages)
        return current > int(self.budget_tokens * self.threshold)

    def usage_fraction(self, messages: list[AgentMessage]) -> float:
        """Return fraction of budget used (0.0 to 1.0+)."""
        if self.budget_tokens <= 0:
            return 1.0
        return self.estimate_tokens(messages) / self.budget_tokens

    async def compact(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """Compact messages to fit within the budget."""
        return await self.compactor.compact(messages, self.budget_tokens)


# ---------------------------------------------------------------------------
# Tool-result truncation
# ---------------------------------------------------------------------------


def _truncate_text(text: str, max_chars: int, *, marker: str) -> str:
    """Truncate text in the middle, preserving head + tail."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if len(marker) >= max_chars:
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:] if tail > 0 else text[:head] + marker


class ToolResultTruncator(ContextCompactor):
    """
    Truncate large tool-result messages to a per-result character ceiling.

    The most recent ``keep_recent`` tool-result messages are left untouched so
    the model can act on fresh data. Older tool results are truncated in the
    middle (head + tail preserved) with an elision marker. Non-tool messages
    are never modified.

    This compactor mutates copies of messages; the input list is not modified.
    """

    def __init__(
        self,
        max_chars: int = 4000,
        keep_recent: int = 3,
        marker: str = "\n\n…[truncated]…\n\n",
    ) -> None:
        if max_chars < 32:
            raise ValueError("max_chars must be >= 32")
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        self.max_chars = max_chars
        self.keep_recent = keep_recent
        self.marker = marker

    def _truncate_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return _truncate_text(content, self.max_chars, marker=self.marker)
        if isinstance(content, list):
            out: list[Any] = []
            for block in content:
                if isinstance(block, TextContent):
                    out.append(
                        TextContent(
                            text=_truncate_text(block.text, self.max_chars, marker=self.marker)
                        )
                    )
                else:
                    out.append(block)
            return out
        return content

    async def compact(
        self,
        messages: list[AgentMessage],
        budget_tokens: int,
    ) -> list[AgentMessage]:
        # Identify indices of tool-result messages.
        tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
        if not tool_indices:
            return list(messages)

        # Keep the trailing ``keep_recent`` tool results intact.
        protected = set(tool_indices[-self.keep_recent :]) if self.keep_recent else set()

        out: list[AgentMessage] = []
        for i, msg in enumerate(messages):
            if msg.role == "tool" and i not in protected:
                new_content = self._truncate_content(msg.content)
                if new_content is msg.content:
                    out.append(msg)
                else:
                    out.append(replace(msg, content=new_content))
            else:
                out.append(msg)
        return out


# ---------------------------------------------------------------------------
# Summarizing compactor
# ---------------------------------------------------------------------------


def _format_message_for_summary(msg: AgentMessage) -> str:
    """Render a single message as plain text suitable for an LLM summarizer."""
    role = msg.role.upper()
    body = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
    if msg.tool_calls:
        names = ", ".join(tc.get("name", "?") for tc in msg.tool_calls)
        body = f"{body}\n[tool_calls: {names}]" if body else f"[tool_calls: {names}]"
    if msg.name:
        return f"{role}({msg.name}): {body}".strip()
    return f"{role}: {body}".strip()


class SummarizingCompactor(ContextCompactor):
    """
    Compact oldest messages into a single summary, keep recent tail intact.

    When the message list exceeds ``budget_tokens``, this compactor calls
    ``summarizer`` on a prefix of older messages and replaces them with a
    single ``system`` message containing the summary. The last
    ``keep_recent`` messages are always preserved verbatim.

    The summarizer callable accepts a list of messages and returns a string
    (sync or async). If it raises, the compactor falls back to dropping
    the oldest messages — never crashes the agent loop.

    Example:
        async def llm_summarize(msgs):
            return await client.summarize(msgs)

        compactor = SummarizingCompactor(
            summarizer=llm_summarize,
            keep_recent=6,
        )
    """

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        keep_recent: int = 6,
        summary_role: str = "system",
        summary_prefix: str = "[conversation-summary]\n",
    ) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be >= 1")
        self.summarizer = summarizer or _default_summarizer
        self.keep_recent = keep_recent
        self.summary_role = summary_role
        self.summary_prefix = summary_prefix

    async def _run_summarizer(self, msgs: list[AgentMessage]) -> str:
        try:
            result = self.summarizer(msgs)
            if inspect.isawaitable(result):
                result = await result
            return str(result) if result is not None else ""
        except Exception as exc:  # pragma: no cover - defensive
            return f"(summary unavailable: {type(exc).__name__})"

    async def compact(
        self,
        messages: list[AgentMessage],
        budget_tokens: int,
    ) -> list[AgentMessage]:
        from skillengine.agent import AgentMessage as _AgentMessage  # local import avoids cycle

        if not messages or estimate_messages_tokens(messages) <= budget_tokens:
            return list(messages)

        if len(messages) <= self.keep_recent:
            return list(messages)

        head = messages[: -self.keep_recent]
        tail = messages[-self.keep_recent :]

        summary_text = await self._run_summarizer(head)
        summary_msg = _AgentMessage(
            role=self.summary_role,
            content=f"{self.summary_prefix}{summary_text}".rstrip(),
            metadata={"compacted": True, "compacted_messages": len(head)},
        )

        compacted = [summary_msg, *tail]

        # If still over budget, fall back to dropping oldest tail messages but
        # always preserve the synthesized summary at index 0.
        if estimate_messages_tokens(compacted) > budget_tokens and len(tail) > 1:
            fallback = TokenBudgetCompactor()
            summary_tokens = estimate_message_tokens(summary_msg)
            tail_budget = max(1, budget_tokens - summary_tokens)
            trimmed_tail = await fallback.compact(tail, tail_budget)
            compacted = [summary_msg, *trimmed_tail]

        return compacted


def _default_summarizer(messages: list[AgentMessage]) -> str:
    """Naive text summarizer: keep first + last lines per message, truncated."""
    lines: list[str] = []
    for m in messages:
        text = _format_message_for_summary(m)
        if len(text) > 200:
            text = text[:120] + " … " + text[-60:]
        lines.append(text)
    return "\n".join(lines)
