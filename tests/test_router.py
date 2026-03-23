"""Tests for Performance Router."""

import time

import pytest
from pathlib import Path

from skillkit.a2a.agent_card import AgentCard, AgentCardSkill
from skillkit.a2a.registry import AgentRegistry, AgentStats
from skillkit.a2a.router import PerformanceRouter, RoutingConfig
from skillkit.models import Skill, SkillMetadata


def _make_skill(name: str, description: str, tags: list[str] | None = None) -> Skill:
    return Skill(
        name=name,
        description=description,
        content=f"# {name}",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        metadata=SkillMetadata(tags=tags or []),
    )


def _make_registry(*agents: tuple[str, str]) -> AgentRegistry:
    registry = AgentRegistry()
    for name, desc in agents:
        registry.register_skill(_make_skill(name, desc))
    return registry


class TestBasicRouting:
    def test_route_keyword_match(self):
        registry = _make_registry(
            ("weather", "Query weather forecasts"),
            ("news", "Read latest news articles"),
        )
        router = PerformanceRouter(registry)

        result = router.route("weather forecast today")
        assert result is not None
        assert result.agent.card.name == "weather"

    def test_route_no_match(self):
        registry = _make_registry(("weather", "Query weather"))
        router = PerformanceRouter(registry)

        result = router.route("cook pasta recipe")
        assert result is None

    def test_route_empty_query(self):
        registry = _make_registry(("test", "Test agent"))
        router = PerformanceRouter(registry)

        assert router.route("") is None

    def test_route_empty_registry(self):
        router = PerformanceRouter(AgentRegistry())
        assert router.route("anything") is None

    def test_route_fallbacks(self):
        registry = _make_registry(
            ("weather-pro", "Professional weather forecast service"),
            ("weather-basic", "Basic weather data query"),
            ("news", "News articles"),
        )
        router = PerformanceRouter(registry)

        result = router.route("weather forecast", top_k=3)
        assert result is not None
        assert len(result.fallbacks) >= 1
        # Both weather agents should be in candidates
        all_names = [result.agent.card.name] + [f.card.name for f in result.fallbacks]
        assert "weather-pro" in all_names
        assert "weather-basic" in all_names

    def test_route_exclude(self):
        registry = _make_registry(
            ("weather-a", "Weather service A"),
            ("weather-b", "Weather service B"),
        )
        router = PerformanceRouter(registry)

        result = router.route("weather", exclude=["weather-a"])
        assert result is not None
        assert result.agent.card.name == "weather-b"


class TestPerformanceWeighting:
    def test_high_success_agent_preferred(self):
        registry = _make_registry(
            ("agent-good", "Data analysis service"),
            ("agent-bad", "Data analysis service"),
        )
        router = PerformanceRouter(registry)

        # Give agent-good a great track record
        good = registry.get("agent-good")
        for _ in range(10):
            good.stats.record_success(100.0)

        # Give agent-bad a poor track record
        bad = registry.get("agent-bad")
        for _ in range(10):
            bad.stats.record_failure("error", 100.0)

        result = router.route("data analysis")
        assert result is not None
        assert result.agent.card.name == "agent-good"

    def test_fast_agent_preferred(self):
        registry = _make_registry(
            ("fast", "Processing service"),
            ("slow", "Processing service"),
        )
        router = PerformanceRouter(registry)

        fast = registry.get("fast")
        for _ in range(5):
            fast.stats.record_success(50.0)  # 50ms avg

        slow = registry.get("slow")
        for _ in range(5):
            slow.stats.record_success(10000.0)  # 10s avg

        result = router.route("processing")
        assert result is not None
        assert result.agent.card.name == "fast"

    def test_neutral_without_enough_data(self):
        registry = _make_registry(
            ("new-agent", "Service"),
            ("veteran", "Service"),
        )
        router = PerformanceRouter(
            registry,
            config=RoutingConfig(min_calls_for_stats=5),
        )

        # new-agent has 1 call (below threshold)
        registry.get("new-agent").stats.record_success(100.0)

        # veteran has 5 calls
        vet = registry.get("veteran")
        for _ in range(5):
            vet.stats.record_success(100.0)

        # Both should be viable (new agent isn't penalized)
        result = router.route("service")
        assert result is not None


class TestCostWeighting:
    def test_cheaper_agent_preferred_equal_quality(self):
        registry = AgentRegistry()

        cheap_skill = _make_skill("cheap", "Analysis service")
        cheap_skill._a2a_config = {"cost_hint": "low"}
        registry.register_skill(cheap_skill)

        expensive_skill = _make_skill("expensive", "Analysis service")
        expensive_skill._a2a_config = {"cost_hint": "high"}
        registry.register_skill(expensive_skill)

        # Give equal performance
        for name in ["cheap", "expensive"]:
            agent = registry.get(name)
            for _ in range(5):
                agent.stats.record_success(100.0)

        router = PerformanceRouter(registry)
        result = router.route("analysis")
        assert result is not None
        assert result.agent.card.name == "cheap"


class TestOutcomeRecording:
    def test_record_success(self):
        registry = _make_registry(("test", "Test agent"))
        router = PerformanceRouter(registry)

        router.record_outcome("test query", "test", True, 150.0)

        agent = registry.get("test")
        assert agent.stats.total_calls == 1
        assert agent.stats.success_rate == 1.0
        assert agent.stats.avg_latency_ms == 150.0

    def test_record_failure(self):
        registry = _make_registry(("test", "Test agent"))
        router = PerformanceRouter(registry)

        router.record_outcome("test query", "test", False, 5000.0)

        agent = registry.get("test")
        assert agent.stats.failure_count == 1

    def test_history_bounded(self):
        registry = _make_registry(("test", "Test agent"))
        router = PerformanceRouter(registry)
        router._max_history = 5

        for i in range(10):
            router.record_outcome(f"query {i}", "test", True, 100.0)

        assert len(router._history) == 5


class TestFallbackExecution:
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        registry = _make_registry(
            ("primary", "Service primary"),
            ("backup", "Service backup"),
        )
        router = PerformanceRouter(registry)

        call_count = 0

        async def execute(agent):
            nonlocal call_count
            call_count += 1
            if agent.card.name == "primary":
                raise Exception("primary failed")
            return "backup success"

        output, winner = await router.route_with_fallback("service", execute)
        assert output == "backup success"
        assert winner.card.name == "backup"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_fallback_needed(self):
        registry = _make_registry(("primary", "Service"))
        router = PerformanceRouter(registry)

        async def execute(agent):
            return "ok"

        output, winner = await router.route_with_fallback("service", execute)
        assert output == "ok"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        registry = _make_registry(("a", "Service agent"), ("b", "Service agent"))
        router = PerformanceRouter(registry)

        async def execute(agent):
            raise Exception("always fails")

        with pytest.raises(RuntimeError, match="All .* agents failed"):
            await router.route_with_fallback("service agent", execute)

    @pytest.mark.asyncio
    async def test_no_match_raises(self):
        router = PerformanceRouter(AgentRegistry())

        async def execute(agent):
            return "ok"

        with pytest.raises(RuntimeError, match="No agents match"):
            await router.route_with_fallback("anything", execute)


class TestRoutingReport:
    def test_report_structure(self):
        registry = _make_registry(
            ("weather", "Weather service"),
            ("news", "News service"),
        )
        registry.get("weather").stats.record_success(100.0)

        router = PerformanceRouter(registry)
        report = router.routing_report("weather forecast")

        assert report["query"] == "weather forecast"
        assert len(report["candidates"]) == 2

        # Find weather candidate
        weather = next(c for c in report["candidates"] if c["name"] == "weather")
        assert "total_score" in weather
        assert "breakdown" in weather
        assert "keyword" in weather["breakdown"]
        assert "performance" in weather["breakdown"]
        assert "cost" in weather["breakdown"]
        assert "recency" in weather["breakdown"]
        assert weather["stats"]["calls"] == 1

    def test_report_empty_query(self):
        router = PerformanceRouter(AgentRegistry())
        report = router.routing_report("")
        assert report["candidates"] == []
