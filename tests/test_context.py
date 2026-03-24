"""Tests for the context management module."""

import asyncio

import pytest

from skillengine.agent import AgentMessage
from skillengine.context import (
    ContextManager,
    SlidingWindowCompactor,
    TokenBudgetCompactor,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello") >= 1
        assert estimate_tokens("hello world this is a test") >= 1

    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_proportional(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("a" * 400)
        assert long > short

    def test_chars_div_4(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25


class TestEstimateMessageTokens:
    def test_simple_message(self):
        msg = AgentMessage(role="user", content="Hello, world!")
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_message_with_reasoning(self):
        msg_without = AgentMessage(role="assistant", content="Answer")
        msg_with = AgentMessage(role="assistant", content="Answer", reasoning="Let me think...")
        assert estimate_message_tokens(msg_with) > estimate_message_tokens(msg_without)

    def test_message_with_tool_calls(self):
        msg = AgentMessage(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "tc_1", "name": "execute", "arguments": '{"command": "ls -la"}'},
            ],
        )
        tokens = estimate_message_tokens(msg)
        assert tokens > 4  # More than just overhead

    def test_estimate_messages_tokens(self):
        messages = [
            AgentMessage(role="user", content="Hello"),
            AgentMessage(role="assistant", content="Hi there!"),
        ]
        total = estimate_messages_tokens(messages)
        individual = sum(estimate_message_tokens(m) for m in messages)
        assert total == individual

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0


# ---------------------------------------------------------------------------
# SlidingWindowCompactor
# ---------------------------------------------------------------------------


class TestSlidingWindowCompactor:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_no_compaction_needed(self):
        compactor = SlidingWindowCompactor(max_turns=10)
        messages = [
            AgentMessage(role="user", content="Hello"),
            AgentMessage(role="assistant", content="Hi!"),
        ]
        result = self._run(compactor.compact(messages, budget_tokens=100_000))
        assert len(result) == 2

    def test_trim_to_max_turns(self):
        compactor = SlidingWindowCompactor(max_turns=2)
        messages = [
            # Turn 1
            AgentMessage(role="user", content="First"),
            AgentMessage(role="assistant", content="Response 1"),
            # Turn 2
            AgentMessage(role="user", content="Second"),
            AgentMessage(role="assistant", content="Response 2"),
            # Turn 3
            AgentMessage(role="user", content="Third"),
            AgentMessage(role="assistant", content="Response 3"),
        ]
        result = self._run(compactor.compact(messages, budget_tokens=100_000))
        # Should keep last 2 turns
        assert len(result) == 4
        assert result[0].content == "Second"
        assert result[-1].content == "Response 3"

    def test_empty_messages(self):
        compactor = SlidingWindowCompactor(max_turns=5)
        result = self._run(compactor.compact([], budget_tokens=100_000))
        assert result == []

    def test_with_tool_messages(self):
        compactor = SlidingWindowCompactor(max_turns=1)
        messages = [
            # Turn 1
            AgentMessage(role="user", content="Old"),
            AgentMessage(role="assistant", content="", tool_calls=[{"id": "1", "name": "exec", "arguments": "{}"}]),
            AgentMessage(role="tool", content="result", tool_call_id="1"),
            AgentMessage(role="assistant", content="Done"),
            # Turn 2
            AgentMessage(role="user", content="Recent"),
            AgentMessage(role="assistant", content="Sure!"),
        ]
        result = self._run(compactor.compact(messages, budget_tokens=100_000))
        # Should keep only last turn
        assert len(result) == 2
        assert result[0].content == "Recent"


# ---------------------------------------------------------------------------
# TokenBudgetCompactor
# ---------------------------------------------------------------------------


class TestTokenBudgetCompactor:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_no_compaction_needed(self):
        compactor = TokenBudgetCompactor()
        messages = [
            AgentMessage(role="user", content="Hello"),
            AgentMessage(role="assistant", content="Hi!"),
        ]
        result = self._run(compactor.compact(messages, budget_tokens=100_000))
        assert len(result) == 2

    def test_compacts_to_fit(self):
        compactor = TokenBudgetCompactor()
        # Create messages that exceed a small budget
        messages = [
            AgentMessage(role="user", content="a" * 400),     # ~100 tokens
            AgentMessage(role="assistant", content="b" * 400), # ~100 tokens
            AgentMessage(role="user", content="c" * 400),      # ~100 tokens
            AgentMessage(role="assistant", content="d" * 400),  # ~100 tokens
        ]
        # Budget for ~250 tokens — should drop oldest
        result = self._run(compactor.compact(messages, budget_tokens=250))
        assert len(result) < len(messages)
        # Last messages should be preserved
        assert result[-1].content == "d" * 400

    def test_empty_messages(self):
        compactor = TokenBudgetCompactor()
        result = self._run(compactor.compact([], budget_tokens=100))
        assert result == []

    def test_keeps_at_least_one(self):
        compactor = TokenBudgetCompactor()
        messages = [
            AgentMessage(role="user", content="a" * 1000),
        ]
        # Very small budget
        result = self._run(compactor.compact(messages, budget_tokens=10))
        assert len(result) >= 1

    def test_first_message_is_user(self):
        compactor = TokenBudgetCompactor()
        messages = [
            AgentMessage(role="user", content="a" * 200),
            AgentMessage(role="assistant", content="b" * 200),
            AgentMessage(role="tool", content="c" * 200, tool_call_id="1"),
            AgentMessage(role="user", content="d" * 200),
            AgentMessage(role="assistant", content="e" * 200),
        ]
        result = self._run(compactor.compact(messages, budget_tokens=200))
        # First message should be user or system
        if result:
            assert result[0].role in ("user", "system")


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_budget_tokens(self):
        mgr = ContextManager(context_window=100_000, reserve_tokens=4096)
        assert mgr.budget_tokens == 100_000 - 4096

    def test_should_compact_false(self):
        mgr = ContextManager(context_window=100_000, reserve_tokens=4096)
        messages = [AgentMessage(role="user", content="Hello")]
        assert mgr.should_compact(messages) is False

    def test_should_compact_true(self):
        # Very small window
        mgr = ContextManager(context_window=100, reserve_tokens=10, threshold=0.9)
        # Budget = 90, threshold = 81
        messages = [AgentMessage(role="user", content="a" * 400)]  # ~100 tokens
        assert mgr.should_compact(messages) is True

    def test_estimate_tokens(self):
        mgr = ContextManager()
        messages = [
            AgentMessage(role="user", content="Hello"),
            AgentMessage(role="assistant", content="World"),
        ]
        tokens = mgr.estimate_tokens(messages)
        assert tokens > 0

    def test_usage_fraction(self):
        mgr = ContextManager(context_window=1000, reserve_tokens=0)
        messages = [AgentMessage(role="user", content="a" * 400)]  # ~100 tokens
        fraction = mgr.usage_fraction(messages)
        assert 0.0 < fraction < 1.0

    def test_compact(self):
        mgr = ContextManager(
            context_window=200,
            reserve_tokens=50,
            compactor=TokenBudgetCompactor(),
        )
        messages = [
            AgentMessage(role="user", content="a" * 400),
            AgentMessage(role="assistant", content="b" * 400),
            AgentMessage(role="user", content="c" * 400),
            AgentMessage(role="assistant", content="d" * 400),
        ]
        result = self._run(mgr.compact(messages))
        assert len(result) <= len(messages)

    def test_with_sliding_window(self):
        mgr = ContextManager(
            context_window=100_000,
            compactor=SlidingWindowCompactor(max_turns=2),
        )
        messages = [
            AgentMessage(role="user", content="Turn 1"),
            AgentMessage(role="assistant", content="Resp 1"),
            AgentMessage(role="user", content="Turn 2"),
            AgentMessage(role="assistant", content="Resp 2"),
            AgentMessage(role="user", content="Turn 3"),
            AgentMessage(role="assistant", content="Resp 3"),
        ]
        result = self._run(mgr.compact(messages))
        assert len(result) == 4
        assert result[0].content == "Turn 2"


# ---------------------------------------------------------------------------
# AgentRunner integration
# ---------------------------------------------------------------------------


class TestAgentRunnerContextIntegration:
    """Test that AgentRunner uses ContextManager and ModelRegistry."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_runner(self, **kwargs):
        from unittest.mock import MagicMock

        from skillengine.agent import AgentConfig, AgentRunner

        mock_engine = MagicMock()
        mock_engine.get_snapshot.return_value = MagicMock(
            skills=[], prompt="", version=1
        )

        config = AgentConfig(
            model="test-model",
            base_url="http://localhost",
            api_key="test-key",
            enable_reasoning=False,
        )

        return AgentRunner(engine=mock_engine, config=config, **kwargs)

    def test_model_definition_property(self):
        from skillengine.model_registry import ModelDefinition, ModelRegistry

        reg = ModelRegistry()
        reg.register(
            ModelDefinition(
                id="test-model",
                provider="test",
                context_window=32_000,
            )
        )

        runner = self._make_runner(model_registry=reg)
        md = runner.model_definition
        assert md is not None
        assert md.id == "test-model"
        assert md.context_window == 32_000

    def test_model_definition_none_without_registry(self):
        runner = self._make_runner()
        assert runner.model_definition is None

    def test_model_definition_none_unknown_model(self):
        from skillengine.model_registry import ModelRegistry

        reg = ModelRegistry()
        runner = self._make_runner(model_registry=reg)
        assert runner.model_definition is None

    def test_cumulative_usage(self):
        from skillengine.model_registry import TokenUsage

        runner = self._make_runner()
        assert runner.cumulative_usage.total_tokens == 0

        runner._cumulative_usage += TokenUsage(input_tokens=100, output_tokens=50)
        assert runner.cumulative_usage.total_tokens == 150

    def test_reset_usage(self):
        from skillengine.model_registry import TokenUsage

        runner = self._make_runner()
        runner._cumulative_usage += TokenUsage(input_tokens=100)
        runner.reset_usage()
        assert runner.cumulative_usage.total_tokens == 0

    def test_get_context_usage(self):
        from skillengine.model_registry import ModelDefinition, ModelRegistry

        reg = ModelRegistry()
        reg.register(
            ModelDefinition(id="test-model", provider="test", context_window=100_000)
        )

        runner = self._make_runner(model_registry=reg)
        runner._conversation = [
            AgentMessage(role="user", content="Hello"),
            AgentMessage(role="assistant", content="Hi there!"),
        ]

        info = runner.get_context_usage()
        assert info["estimated_tokens"] > 0
        assert info["context_window"] == 100_000
        assert 0.0 < info["usage_fraction"] < 1.0
        assert info["needs_compaction"] is False

    def test_get_context_usage_with_context_manager(self):
        ctx_mgr = ContextManager(context_window=50_000)

        runner = self._make_runner(context_manager=ctx_mgr)
        runner._conversation = [AgentMessage(role="user", content="Hello")]

        info = runner.get_context_usage()
        assert info["context_window"] == 50_000

    def test_context_compaction_in_chat(self):
        """Test that context manager is called during chat()."""
        from unittest.mock import AsyncMock, MagicMock

        runner = self._make_runner(
            context_manager=ContextManager(
                context_window=50,  # Very small
                reserve_tokens=10,
                threshold=0.5,
            ),
        )

        # Mock _call_llm
        runner._call_llm = AsyncMock(
            return_value=AgentMessage(role="assistant", content="Response")
        )

        # Add enough messages that compaction triggers
        runner._conversation = [
            AgentMessage(role="user", content="a" * 400),
            AgentMessage(role="assistant", content="b" * 400),
        ]

        # This should trigger compaction before LLM call
        result = self._run(runner.chat("new question"))
        assert result.content == "Response"
        # Compaction happened — _call_llm was called with potentially fewer messages
        assert runner._call_llm.called
