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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillengine.agent import AgentMessage


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


def estimate_message_tokens(message: AgentMessage) -> int:
    """
    Estimate token count for a single AgentMessage.

    Accounts for role overhead, content, tool call arguments, and reasoning.
    """
    # Base overhead per message (role, separators)
    tokens = 4

    tokens += estimate_tokens(message.content)

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
