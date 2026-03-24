"""
Built-in model catalog.

Contains definitions for major LLM providers. All prices are in $/million tokens.
Data is current as of 2025-05. Update as needed when providers change pricing
or release new models.
"""

from __future__ import annotations

from skillengine.model_registry import ModelCost, ModelDefinition


def get_default_models() -> list[ModelDefinition]:
    """Return the built-in model definitions."""
    return [
        # ---------------------------------------------------------------
        # Anthropic
        # ---------------------------------------------------------------
        ModelDefinition(
            id="claude-opus-4-20250514",
            provider="anthropic",
            api="anthropic",
            display_name="Claude Opus 4",
            context_window=200_000,
            max_output_tokens=32_000,
            cost=ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="claude-sonnet-4-20250514",
            provider="anthropic",
            api="anthropic",
            display_name="Claude Sonnet 4",
            context_window=200_000,
            max_output_tokens=16_000,
            cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="claude-haiku-4-20250414",
            provider="anthropic",
            api="anthropic",
            display_name="Claude Haiku 4",
            context_window=200_000,
            max_output_tokens=8_192,
            cost=ModelCost(input=0.80, output=4.0, cache_read=0.08, cache_write=1.0),
            capabilities={"text", "image", "tool_use"},
            reasoning=False,
            input_modalities=["text", "image"],
        ),
        # ---------------------------------------------------------------
        # OpenAI
        # ---------------------------------------------------------------
        ModelDefinition(
            id="gpt-4o",
            provider="openai",
            api="openai",
            display_name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            cost=ModelCost(input=2.50, output=10.0, cache_read=1.25),
            capabilities={"text", "image", "tool_use"},
            reasoning=False,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="gpt-4o-mini",
            provider="openai",
            api="openai",
            display_name="GPT-4o Mini",
            context_window=128_000,
            max_output_tokens=16_384,
            cost=ModelCost(input=0.15, output=0.60, cache_read=0.075),
            capabilities={"text", "image", "tool_use"},
            reasoning=False,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="o3",
            provider="openai",
            api="openai",
            display_name="o3",
            context_window=200_000,
            max_output_tokens=100_000,
            cost=ModelCost(input=10.0, output=40.0, cache_read=2.50),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="o3-mini",
            provider="openai",
            api="openai",
            display_name="o3-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            cost=ModelCost(input=1.10, output=4.40, cache_read=0.55),
            capabilities={"text", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text"],
        ),
        ModelDefinition(
            id="o4-mini",
            provider="openai",
            api="openai",
            display_name="o4-mini",
            context_window=200_000,
            max_output_tokens=100_000,
            cost=ModelCost(input=1.10, output=4.40, cache_read=0.275),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        # ---------------------------------------------------------------
        # Google
        # ---------------------------------------------------------------
        ModelDefinition(
            id="gemini-2.5-pro",
            provider="google",
            api="openai",
            display_name="Gemini 2.5 Pro",
            context_window=1_048_576,
            max_output_tokens=65_536,
            cost=ModelCost(input=1.25, output=10.0),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        ModelDefinition(
            id="gemini-2.5-flash",
            provider="google",
            api="openai",
            display_name="Gemini 2.5 Flash",
            context_window=1_048_576,
            max_output_tokens=65_536,
            cost=ModelCost(input=0.15, output=0.60),
            capabilities={"text", "image", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text", "image"],
        ),
        # ---------------------------------------------------------------
        # DeepSeek
        # ---------------------------------------------------------------
        ModelDefinition(
            id="deepseek-chat",
            provider="deepseek",
            api="openai",
            display_name="DeepSeek V3",
            context_window=64_000,
            max_output_tokens=8_192,
            cost=ModelCost(input=0.27, output=1.10, cache_read=0.07),
            capabilities={"text", "tool_use"},
            reasoning=False,
            input_modalities=["text"],
        ),
        ModelDefinition(
            id="deepseek-reasoner",
            provider="deepseek",
            api="openai",
            display_name="DeepSeek R1",
            context_window=64_000,
            max_output_tokens=8_192,
            cost=ModelCost(input=0.55, output=2.19, cache_read=0.14),
            capabilities={"text", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text"],
        ),
        # ---------------------------------------------------------------
        # MiniMax
        # ---------------------------------------------------------------
        ModelDefinition(
            id="MiniMax-M1",
            provider="minimax",
            api="openai",
            display_name="MiniMax M1",
            context_window=1_000_000,
            max_output_tokens=80_000,
            cost=ModelCost(input=0.20, output=1.10),
            capabilities={"text", "tool_use", "reasoning"},
            reasoning=True,
            input_modalities=["text"],
        ),
    ]
