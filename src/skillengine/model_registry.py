"""
Model metadata and registry.

Provides typed model definitions with cost, context window, and capability
information. The ModelRegistry allows registering built-in and custom models,
looking them up by ID, and calculating costs from token usage.

Example:
    from skillengine.model_registry import ModelRegistry, ModelDefinition, ModelCost

    registry = ModelRegistry()
    registry.load_defaults()  # Load built-in model catalog

    model = registry.get("claude-sonnet-4-20250514")
    print(model.context_window)  # 200000
    print(model.cost.input)      # 3.0 ($/million tokens)

    # Calculate cost from usage
    cost = registry.calculate_cost("claude-sonnet-4-20250514", usage)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Thinking budget levels
# ---------------------------------------------------------------------------

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

DEFAULT_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 16384,
}

# ---------------------------------------------------------------------------
# Transport types
# ---------------------------------------------------------------------------

Transport = Literal["sse", "websocket", "auto"]


@dataclass
class ModelCost:
    """Pricing per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class TokenUsage:
    """Token counts for a single LLM request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.thinking_tokens
        )

    def calculate_cost(self, cost: ModelCost) -> CostBreakdown:
        """Calculate dollar cost from pricing."""
        input_cost = (cost.input / 1_000_000) * self.input_tokens
        output_cost = (cost.output / 1_000_000) * self.output_tokens
        cache_read_cost = (cost.cache_read / 1_000_000) * self.cache_read_tokens
        cache_write_cost = (cost.cache_write / 1_000_000) * self.cache_write_tokens
        return CostBreakdown(
            input=input_cost,
            output=output_cost,
            cache_read=cache_read_cost,
            cache_write=cache_write_cost,
            total=input_cost + output_cost + cache_read_cost + cache_write_cost,
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.thinking_tokens += other.thinking_tokens
        return self


@dataclass
class CostBreakdown:
    """Dollar cost breakdown for a request."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class ModelDefinition:
    """
    Metadata for an LLM model.

    Attributes:
        id: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        provider: Provider name (e.g., "openai", "anthropic", "minimax").
        api: API protocol (e.g., "openai", "anthropic"). Used to select adapter.
        display_name: Human-readable name for UI display.
        context_window: Maximum input tokens the model accepts.
        max_output_tokens: Maximum tokens the model can generate.
        cost: Pricing per million tokens.
        capabilities: Feature set (e.g., {"text", "image", "tool_use", "reasoning"}).
        reasoning: Whether the model supports extended thinking / chain-of-thought.
        input_modalities: Supported input types (e.g., ["text", "image"]).
    """

    id: str
    provider: str
    api: str = "openai"
    display_name: str = ""
    context_window: int = 128_000
    max_output_tokens: int = 4096
    cost: ModelCost = field(default_factory=ModelCost)
    capabilities: set[str] = field(default_factory=lambda: {"text", "tool_use"})
    reasoning: bool = False
    input_modalities: list[str] = field(default_factory=lambda: ["text"])

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id

    def supports(self, capability: str) -> bool:
        """Check if this model supports a given capability."""
        return capability in self.capabilities


class ModelRegistry:
    """
    Registry of model definitions.

    Supports registering models individually or loading the built-in catalog.
    Models can be looked up by ID, filtered by provider, and used for cost
    calculations.

    Example:
        registry = ModelRegistry()
        registry.load_defaults()

        # Look up
        model = registry.get("gpt-4o")

        # Filter
        anthropic_models = registry.list_by_provider("anthropic")

        # Cost
        cost = registry.calculate_cost("gpt-4o", usage)
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelDefinition] = {}

    def register(self, model: ModelDefinition) -> None:
        """Register a model definition. Overwrites any existing entry with the same ID."""
        self._models[model.id] = model

    def unregister(self, model_id: str) -> bool:
        """Remove a model by ID. Returns True if it existed."""
        return self._models.pop(model_id, None) is not None

    def get(self, model_id: str) -> ModelDefinition | None:
        """Get a model by exact ID."""
        return self._models.get(model_id)

    def find(self, query: str) -> list[ModelDefinition]:
        """Find models whose ID or display_name contains the query (case-insensitive)."""
        q = query.lower()
        return [
            m for m in self._models.values() if q in m.id.lower() or q in m.display_name.lower()
        ]

    def list_by_provider(self, provider: str) -> list[ModelDefinition]:
        """List all models from a given provider."""
        return [m for m in self._models.values() if m.provider == provider]

    def list_by_capability(self, capability: str) -> list[ModelDefinition]:
        """List all models that support a given capability."""
        return [m for m in self._models.values() if m.supports(capability)]

    def all(self) -> list[ModelDefinition]:
        """Return all registered models."""
        return list(self._models.values())

    @property
    def count(self) -> int:
        return len(self._models)

    def calculate_cost(self, model_id: str, usage: TokenUsage) -> CostBreakdown:
        """Calculate cost for a model given token usage. Returns zero cost if model not found."""
        model = self.get(model_id)
        if model is None:
            return CostBreakdown()
        return usage.calculate_cost(model.cost)

    def load_defaults(self) -> int:
        """
        Load built-in model definitions from the catalog.

        Returns the number of models loaded.
        """
        from skillengine.models_catalog import get_default_models

        models = get_default_models()
        for model in models:
            self.register(model)
        return len(models)

    def load_from_dicts(self, model_dicts: list[dict[str, Any]]) -> int:
        """
        Load models from a list of dictionaries (e.g., from JSON/YAML config).

        Each dict should have keys matching ModelDefinition fields.
        Returns the number of models loaded.
        """
        count = 0
        for d in model_dicts:
            cost_data = d.get("cost", {})
            cost = ModelCost(
                input=cost_data.get("input", 0.0),
                output=cost_data.get("output", 0.0),
                cache_read=cost_data.get("cache_read", 0.0),
                cache_write=cost_data.get("cache_write", 0.0),
            )
            caps = d.get("capabilities", ["text", "tool_use"])
            model = ModelDefinition(
                id=d["id"],
                provider=d.get("provider", ""),
                api=d.get("api", "openai"),
                display_name=d.get("display_name", ""),
                context_window=d.get("context_window", 128_000),
                max_output_tokens=d.get("max_output_tokens", 4096),
                cost=cost,
                capabilities=set(caps) if isinstance(caps, list) else caps,
                reasoning=d.get("reasoning", False),
                input_modalities=d.get("input_modalities", ["text"]),
            )
            self.register(model)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Thinking budget helpers
# ---------------------------------------------------------------------------

_ADAPTIVE_PATTERN = re.compile(r"opus[_-]4[_.-]6|opus-4-6")


def supports_adaptive_thinking(model_id: str) -> bool:
    """Return True if the model supports adaptive thinking (effort-based).

    Currently only Opus 4.6 variants support adaptive thinking.
    """
    return bool(_ADAPTIVE_PATTERN.search(model_id.lower()))


def map_thinking_level_to_anthropic_effort(
    level: ThinkingLevel,
) -> Literal["low", "medium", "high", "max"]:
    """Map a ThinkingLevel to Anthropic's effort parameter."""
    mapping: dict[ThinkingLevel, Literal["low", "medium", "high", "max"]] = {
        "off": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    }
    return mapping[level]


def map_thinking_level_to_openai_effort(
    level: ThinkingLevel,
) -> Literal["low", "medium", "high"]:
    """Map a ThinkingLevel to OpenAI's reasoning_effort parameter."""
    mapping: dict[ThinkingLevel, Literal["low", "medium", "high"]] = {
        "off": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }
    return mapping[level]


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    level: ThinkingLevel,
    custom_budgets: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Calculate max_tokens and thinking budget for a given thinking level.

    Returns:
        (max_tokens, thinking_budget) where max_tokens includes the thinking
        budget and thinking_budget is the number of tokens reserved for
        thinking. When level is "off", thinking_budget is 0 and max_tokens
        is returned unchanged.
    """
    if level == "off":
        return base_max_tokens, 0

    budgets = custom_budgets if custom_budgets is not None else DEFAULT_THINKING_BUDGETS
    thinking_budget = budgets.get(level, DEFAULT_THINKING_BUDGETS.get(level, 8192))

    # max_tokens must include the thinking budget and fit within model limits
    required = base_max_tokens + thinking_budget
    max_tokens = min(required, model_max_tokens)

    # Ensure thinking budget doesn't exceed the clamped max_tokens
    thinking_budget = min(thinking_budget, max_tokens - 1)

    return max_tokens, thinking_budget
