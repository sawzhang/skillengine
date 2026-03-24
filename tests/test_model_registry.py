"""Tests for model_registry module."""

import pytest

from skillengine.model_registry import (
    CostBreakdown,
    ModelCost,
    ModelDefinition,
    ModelRegistry,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=20)
        assert usage.total_tokens == 170

    def test_total_tokens_zero(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0

    def test_calculate_cost(self):
        cost = ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_read_tokens=200_000,
            cache_write_tokens=100_000,
        )
        breakdown = usage.calculate_cost(cost)
        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.input == pytest.approx(3.0)
        assert breakdown.output == pytest.approx(7.5)
        assert breakdown.cache_read == pytest.approx(0.06)
        assert breakdown.cache_write == pytest.approx(0.375)
        assert breakdown.total == pytest.approx(10.935)

    def test_calculate_cost_zero(self):
        cost = ModelCost(input=3.0, output=15.0)
        usage = TokenUsage()
        breakdown = usage.calculate_cost(cost)
        assert breakdown.total == 0.0

    def test_add(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=100, cache_read_tokens=30)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 150
        assert c.cache_read_tokens == 30
        assert c.cache_write_tokens == 0
        # Original unchanged
        assert a.input_tokens == 100

    def test_iadd(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=100)
        a += b
        assert a.input_tokens == 300
        assert a.output_tokens == 150


# ---------------------------------------------------------------------------
# ModelCost
# ---------------------------------------------------------------------------


class TestModelCost:
    def test_defaults(self):
        cost = ModelCost()
        assert cost.input == 0.0
        assert cost.output == 0.0
        assert cost.cache_read == 0.0
        assert cost.cache_write == 0.0

    def test_custom(self):
        cost = ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
        assert cost.input == 3.0
        assert cost.output == 15.0


# ---------------------------------------------------------------------------
# ModelDefinition
# ---------------------------------------------------------------------------


class TestModelDefinition:
    def test_basic(self):
        model = ModelDefinition(id="gpt-4o", provider="openai")
        assert model.id == "gpt-4o"
        assert model.provider == "openai"
        assert model.api == "openai"
        assert model.display_name == "gpt-4o"  # Auto-filled
        assert model.context_window == 128_000
        assert model.reasoning is False

    def test_display_name_auto(self):
        model = ModelDefinition(id="my-model", provider="custom")
        assert model.display_name == "my-model"

    def test_display_name_explicit(self):
        model = ModelDefinition(id="my-model", provider="custom", display_name="My Model")
        assert model.display_name == "My Model"

    def test_supports(self):
        model = ModelDefinition(
            id="test",
            provider="test",
            capabilities={"text", "image", "tool_use"},
        )
        assert model.supports("text") is True
        assert model.supports("image") is True
        assert model.supports("reasoning") is False

    def test_defaults(self):
        model = ModelDefinition(id="test", provider="test")
        assert "text" in model.capabilities
        assert "tool_use" in model.capabilities
        assert model.input_modalities == ["text"]
        assert model.max_output_tokens == 4096


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        model = ModelDefinition(id="gpt-4o", provider="openai")
        reg.register(model)
        assert reg.get("gpt-4o") is model
        assert reg.count == 1

    def test_get_missing(self):
        reg = ModelRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = ModelRegistry()
        model = ModelDefinition(id="gpt-4o", provider="openai")
        reg.register(model)
        assert reg.unregister("gpt-4o") is True
        assert reg.get("gpt-4o") is None
        assert reg.count == 0

    def test_unregister_missing(self):
        reg = ModelRegistry()
        assert reg.unregister("nonexistent") is False

    def test_overwrite(self):
        reg = ModelRegistry()
        model1 = ModelDefinition(id="gpt-4o", provider="openai", context_window=128_000)
        model2 = ModelDefinition(id="gpt-4o", provider="openai", context_window=256_000)
        reg.register(model1)
        reg.register(model2)
        assert reg.get("gpt-4o").context_window == 256_000
        assert reg.count == 1

    def test_find(self):
        reg = ModelRegistry()
        reg.register(ModelDefinition(id="gpt-4o", provider="openai", display_name="GPT-4o"))
        reg.register(ModelDefinition(id="gpt-4o-mini", provider="openai", display_name="GPT-4o Mini"))
        reg.register(ModelDefinition(id="claude-sonnet-4-20250514", provider="anthropic"))

        results = reg.find("gpt")
        assert len(results) == 2

        results = reg.find("claude")
        assert len(results) == 1

        results = reg.find("nonexistent")
        assert len(results) == 0

    def test_find_case_insensitive(self):
        reg = ModelRegistry()
        reg.register(ModelDefinition(id="GPT-4o", provider="openai", display_name="GPT-4o"))
        results = reg.find("gpt")
        assert len(results) == 1

    def test_list_by_provider(self):
        reg = ModelRegistry()
        reg.register(ModelDefinition(id="gpt-4o", provider="openai"))
        reg.register(ModelDefinition(id="gpt-4o-mini", provider="openai"))
        reg.register(ModelDefinition(id="claude-sonnet", provider="anthropic"))

        openai_models = reg.list_by_provider("openai")
        assert len(openai_models) == 2

        anthropic_models = reg.list_by_provider("anthropic")
        assert len(anthropic_models) == 1

    def test_list_by_capability(self):
        reg = ModelRegistry()
        reg.register(
            ModelDefinition(
                id="reasoning-model",
                provider="test",
                capabilities={"text", "reasoning", "tool_use"},
            )
        )
        reg.register(
            ModelDefinition(
                id="basic-model",
                provider="test",
                capabilities={"text", "tool_use"},
            )
        )

        reasoning = reg.list_by_capability("reasoning")
        assert len(reasoning) == 1
        assert reasoning[0].id == "reasoning-model"

    def test_all(self):
        reg = ModelRegistry()
        reg.register(ModelDefinition(id="a", provider="x"))
        reg.register(ModelDefinition(id="b", provider="y"))
        assert len(reg.all()) == 2

    def test_calculate_cost(self):
        reg = ModelRegistry()
        reg.register(
            ModelDefinition(
                id="test-model",
                provider="test",
                cost=ModelCost(input=3.0, output=15.0),
            )
        )
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        breakdown = reg.calculate_cost("test-model", usage)
        assert breakdown.input == pytest.approx(3.0)
        assert breakdown.output == pytest.approx(7.5)
        assert breakdown.total == pytest.approx(10.5)

    def test_calculate_cost_unknown_model(self):
        reg = ModelRegistry()
        usage = TokenUsage(input_tokens=1000)
        breakdown = reg.calculate_cost("nonexistent", usage)
        assert breakdown.total == 0.0

    def test_load_defaults(self):
        reg = ModelRegistry()
        count = reg.load_defaults()
        assert count > 0
        assert reg.count == count
        # Should have some known models
        assert reg.get("gpt-4o") is not None
        assert reg.get("claude-sonnet-4-20250514") is not None

    def test_load_from_dicts(self):
        reg = ModelRegistry()
        dicts = [
            {
                "id": "custom-model",
                "provider": "custom",
                "api": "openai",
                "context_window": 32_000,
                "cost": {"input": 1.0, "output": 5.0},
                "capabilities": ["text", "tool_use", "image"],
                "reasoning": True,
            }
        ]
        count = reg.load_from_dicts(dicts)
        assert count == 1
        model = reg.get("custom-model")
        assert model is not None
        assert model.provider == "custom"
        assert model.context_window == 32_000
        assert model.cost.input == 1.0
        assert model.reasoning is True
        assert "image" in model.capabilities


# ---------------------------------------------------------------------------
# Built-in catalog
# ---------------------------------------------------------------------------


class TestModelsCatalog:
    def test_catalog_loads(self):
        from skillengine.models_catalog import get_default_models

        models = get_default_models()
        assert len(models) > 0
        # All should be ModelDefinition instances
        for m in models:
            assert isinstance(m, ModelDefinition)
            assert m.id
            assert m.provider

    def test_catalog_has_major_providers(self):
        from skillengine.models_catalog import get_default_models

        models = get_default_models()
        providers = {m.provider for m in models}
        assert "openai" in providers
        assert "anthropic" in providers

    def test_catalog_costs_positive(self):
        from skillengine.models_catalog import get_default_models

        models = get_default_models()
        for m in models:
            # At minimum, input and output should be set
            assert m.cost.input >= 0
            assert m.cost.output >= 0
