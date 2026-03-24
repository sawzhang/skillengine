"""Tests for thinking budget levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.model_registry import (
    DEFAULT_THINKING_BUDGETS,
    ThinkingLevel,
    TokenUsage,
    adjust_max_tokens_for_thinking,
    map_thinking_level_to_anthropic_effort,
    map_thinking_level_to_openai_effort,
    supports_adaptive_thinking,
)


# ---------------------------------------------------------------------------
# ThinkingLevel type validation
# ---------------------------------------------------------------------------


class TestThinkingLevel:
    def test_valid_levels(self):
        levels: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]
        assert len(levels) == 6

    def test_off_is_valid(self):
        level: ThinkingLevel = "off"
        assert level == "off"


# ---------------------------------------------------------------------------
# DEFAULT_THINKING_BUDGETS
# ---------------------------------------------------------------------------


class TestDefaultThinkingBudgets:
    def test_has_all_non_off_levels(self):
        for level in ("minimal", "low", "medium", "high", "xhigh"):
            assert level in DEFAULT_THINKING_BUDGETS

    def test_off_not_in_budgets(self):
        assert "off" not in DEFAULT_THINKING_BUDGETS

    def test_values_are_positive(self):
        for value in DEFAULT_THINKING_BUDGETS.values():
            assert value > 0

    def test_expected_values(self):
        assert DEFAULT_THINKING_BUDGETS["minimal"] == 1024
        assert DEFAULT_THINKING_BUDGETS["low"] == 2048
        assert DEFAULT_THINKING_BUDGETS["medium"] == 8192
        assert DEFAULT_THINKING_BUDGETS["high"] == 16384
        assert DEFAULT_THINKING_BUDGETS["xhigh"] == 16384


# ---------------------------------------------------------------------------
# adjust_max_tokens_for_thinking
# ---------------------------------------------------------------------------


class TestAdjustMaxTokensForThinking:
    def test_off_returns_unchanged(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(4096, 128_000, "off")
        assert max_tokens == 4096
        assert budget == 0

    def test_medium_level(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(4096, 128_000, "medium")
        assert budget == 8192
        assert max_tokens == 4096 + 8192

    def test_clamping_to_model_max(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(4096, 5000, "high")
        assert max_tokens == 5000
        # budget should be clamped: min(16384, 5000 - 1) = 4999
        assert budget == 4999

    def test_custom_budgets(self):
        custom = {"low": 512, "medium": 1024}
        max_tokens, budget = adjust_max_tokens_for_thinking(
            4096, 128_000, "low", custom_budgets=custom
        )
        assert budget == 512
        assert max_tokens == 4096 + 512

    def test_custom_budgets_missing_level_falls_back(self):
        custom = {"low": 512}
        max_tokens, budget = adjust_max_tokens_for_thinking(
            4096, 128_000, "high", custom_budgets=custom
        )
        # Falls back to DEFAULT_THINKING_BUDGETS["high"] = 16384
        assert budget == 16384

    def test_all_levels(self):
        for level in ("minimal", "low", "medium", "high", "xhigh"):
            max_tokens, budget = adjust_max_tokens_for_thinking(4096, 128_000, level)
            assert budget > 0
            assert max_tokens > 4096

    def test_edge_tiny_model_max(self):
        max_tokens, budget = adjust_max_tokens_for_thinking(100, 200, "medium")
        assert max_tokens == 200
        assert budget == 199  # min(8192, 200-1)


# ---------------------------------------------------------------------------
# map_thinking_level_to_anthropic_effort
# ---------------------------------------------------------------------------


class TestMapToAnthropicEffort:
    def test_off(self):
        assert map_thinking_level_to_anthropic_effort("off") == "low"

    def test_minimal(self):
        assert map_thinking_level_to_anthropic_effort("minimal") == "low"

    def test_low(self):
        assert map_thinking_level_to_anthropic_effort("low") == "low"

    def test_medium(self):
        assert map_thinking_level_to_anthropic_effort("medium") == "medium"

    def test_high(self):
        assert map_thinking_level_to_anthropic_effort("high") == "high"

    def test_xhigh(self):
        assert map_thinking_level_to_anthropic_effort("xhigh") == "max"


# ---------------------------------------------------------------------------
# map_thinking_level_to_openai_effort
# ---------------------------------------------------------------------------


class TestMapToOpenAIEffort:
    def test_off(self):
        assert map_thinking_level_to_openai_effort("off") == "low"

    def test_minimal(self):
        assert map_thinking_level_to_openai_effort("minimal") == "low"

    def test_low(self):
        assert map_thinking_level_to_openai_effort("low") == "low"

    def test_medium(self):
        assert map_thinking_level_to_openai_effort("medium") == "medium"

    def test_high(self):
        assert map_thinking_level_to_openai_effort("high") == "high"

    def test_xhigh(self):
        assert map_thinking_level_to_openai_effort("xhigh") == "high"


# ---------------------------------------------------------------------------
# supports_adaptive_thinking
# ---------------------------------------------------------------------------


class TestSupportsAdaptiveThinking:
    def test_opus_4_6_hyphen(self):
        assert supports_adaptive_thinking("claude-opus-4-6") is True

    def test_opus_4_dot_6(self):
        assert supports_adaptive_thinking("claude-opus-4.6") is True

    def test_opus_4_6_underscore(self):
        assert supports_adaptive_thinking("claude_opus_4_6") is True

    def test_sonnet_false(self):
        assert supports_adaptive_thinking("claude-sonnet-4-20250514") is False

    def test_gpt_false(self):
        assert supports_adaptive_thinking("gpt-4o") is False


# ---------------------------------------------------------------------------
# TokenUsage.thinking_tokens
# ---------------------------------------------------------------------------


class TestTokenUsageThinking:
    def test_default_zero(self):
        usage = TokenUsage()
        assert usage.thinking_tokens == 0

    def test_total_includes_thinking(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, thinking_tokens=200)
        assert usage.total_tokens == 350

    def test_add_includes_thinking(self):
        a = TokenUsage(thinking_tokens=100)
        b = TokenUsage(thinking_tokens=200)
        result = a + b
        assert result.thinking_tokens == 300

    def test_iadd_includes_thinking(self):
        a = TokenUsage(thinking_tokens=100)
        b = TokenUsage(thinking_tokens=200)
        a += b
        assert a.thinking_tokens == 300


# ---------------------------------------------------------------------------
# AgentConfig.thinking_level
# ---------------------------------------------------------------------------


class TestAgentConfigThinking:
    def test_default_none(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig()
        assert config.thinking_level is None

    def test_set_level(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig(thinking_level="high")
        assert config.thinking_level == "high"

    def test_transport_default(self):
        from skillengine.agent import AgentConfig

        config = AgentConfig()
        assert config.transport == "sse"


# ---------------------------------------------------------------------------
# Anthropic adapter thinking params (mocked)
# ---------------------------------------------------------------------------


try:
    import anthropic  # noqa: F401

    _has_anthropic = True
except ImportError:
    _has_anthropic = False


@pytest.mark.skipif(not _has_anthropic, reason="anthropic not installed")
class TestAnthropicAdapterThinking:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> Any:
        from skillengine import SkillsConfig, SkillsEngine

        return SkillsEngine(config=SkillsConfig(skill_dirs=[]))

    @pytest.mark.asyncio
    async def test_adaptive_thinking_params(self, engine: Any):
        from skillengine.adapters.anthropic import AnthropicAdapter
        from skillengine.adapters.base import Message

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20, thinking_tokens=100
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = AnthropicAdapter(
            engine, client=mock_client, model="claude-opus-4-6", enable_tools=False
        )

        await adapter.chat(
            [Message(role="user", content="hello")],
            thinking_level="high",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_budget_thinking_params(self, engine: Any):
        from skillengine.adapters.anthropic import AnthropicAdapter
        from skillengine.adapters.base import Message

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20
        )
        # No thinking_tokens attribute
        del mock_response.usage.thinking_tokens
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = AnthropicAdapter(
            engine, client=mock_client, model="claude-sonnet-4-20250514", enable_tools=False
        )

        await adapter.chat(
            [Message(role="user", content="hello")],
            thinking_level="medium",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_thinking_tokens_extraction(self, engine: Any):
        from skillengine.adapters.anthropic import AnthropicAdapter
        from skillengine.adapters.base import Message

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=20, thinking_tokens=500
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = AnthropicAdapter(
            engine, client=mock_client, model="claude-opus-4-6", enable_tools=False
        )

        response = await adapter.chat(
            [Message(role="user", content="hello")],
            thinking_level="high",
        )

        assert response.token_usage is not None
        assert response.token_usage.thinking_tokens == 500


# ---------------------------------------------------------------------------
# OpenAI adapter reasoning_effort (mocked)
# ---------------------------------------------------------------------------


class TestOpenAIAdapterThinking:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> Any:
        from skillengine import SkillsConfig, SkillsEngine

        return SkillsEngine(config=SkillsConfig(skill_dirs=[]))

    @pytest.mark.asyncio
    async def test_reasoning_effort_for_o3(self, engine: Any):
        from skillengine.adapters.base import Message
        from skillengine.adapters.openai import OpenAIAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = OpenAIAdapter(
            engine, client=mock_client, model="o3-mini", enable_tools=False
        )

        await adapter.chat(
            [Message(role="user", content="hello")],
            thinking_level="high",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_no_reasoning_effort_for_gpt4(self, engine: Any):
        from skillengine.adapters.base import Message
        from skillengine.adapters.openai import OpenAIAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = OpenAIAdapter(
            engine, client=mock_client, model="gpt-4o", enable_tools=False
        )

        await adapter.chat(
            [Message(role="user", content="hello")],
            thinking_level="high",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs
