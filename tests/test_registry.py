"""Tests for AgentRegistry."""

from pathlib import Path

from skillkit.a2a.agent_card import AgentCard, AgentCardSkill
from skillkit.a2a.registry import AgentRegistry, AgentSource, AgentStats, RegisteredAgent
from skillkit.models import Skill, SkillMetadata


def _make_skill(name: str, description: str, tags: list[str] | None = None) -> Skill:
    return Skill(
        name=name,
        description=description,
        content=f"# {name}\nDo the thing.",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        metadata=SkillMetadata(tags=tags or []),
    )


def _make_card(name: str, description: str, tags: list[str] | None = None) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        tags=tags or [],
        skills=[AgentCardSkill(name=name, description=description, tags=tags or [])],
    )


class TestRegistration:
    def test_register_skill(self):
        registry = AgentRegistry()
        skill = _make_skill("test", "A test skill")
        agent = registry.register_skill(skill)

        assert agent.source == AgentSource.LOCAL
        assert agent.skill is skill
        assert agent.card.name == "test"
        assert registry.count == 1

    def test_register_remote(self):
        registry = AgentRegistry()
        card = _make_card("remote-agent", "A remote agent")
        agent = registry.register_remote(card, endpoint="http://localhost:9000")

        assert agent.source == AgentSource.REMOTE
        assert agent.endpoint == "http://localhost:9000"
        assert agent.skill is None
        assert registry.count == 1

    def test_register_multiple_skills(self):
        registry = AgentRegistry()
        skills = [
            _make_skill("a", "Agent A"),
            _make_skill("b", "Agent B"),
            _make_skill("c", "Agent C"),
        ]
        count = registry.register_skills(skills)

        assert count == 3
        assert registry.count == 3

    def test_unregister(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Test"))

        assert registry.unregister("test") is True
        assert registry.count == 0
        assert registry.unregister("test") is False  # already removed

    def test_overwrite_registration(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Version 1"))
        registry.register_skill(_make_skill("test", "Version 2"))

        assert registry.count == 1
        assert registry.get("test").card.description == "Version 2"

    def test_clear(self):
        registry = AgentRegistry()
        registry.register_skills([_make_skill("a", "A"), _make_skill("b", "B")])
        registry.clear()

        assert registry.count == 0


class TestQueries:
    def test_get_existing(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Test"))

        agent = registry.get("test")
        assert agent is not None
        assert agent.card.name == "test"

    def test_get_missing(self):
        registry = AgentRegistry()
        assert registry.get("nonexistent") is None

    def test_all(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("local", "Local"))
        registry.register_remote(_make_card("remote", "Remote"), "http://x")

        assert len(registry.all()) == 2

    def test_local_agents(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("local", "Local"))
        registry.register_remote(_make_card("remote", "Remote"), "http://x")

        local = registry.local_agents()
        assert len(local) == 1
        assert local[0].source == AgentSource.LOCAL

    def test_remote_agents(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("local", "Local"))
        registry.register_remote(_make_card("remote", "Remote"), "http://x")

        remote = registry.remote_agents()
        assert len(remote) == 1
        assert remote[0].source == AgentSource.REMOTE

    def test_all_cards(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("a", "Agent A"))
        registry.register_skill(_make_skill("b", "Agent B"))

        cards = registry.all_cards()
        assert len(cards) == 2
        names = {c.name for c in cards}
        assert names == {"a", "b"}


class TestCardsSummary:
    def test_empty_registry(self):
        registry = AgentRegistry()
        assert registry.cards_summary() == ""

    def test_summary_content(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("weather", "Query weather", tags=["outdoor"]))
        registry.register_skill(_make_skill("news", "Read latest news"))

        summary = registry.cards_summary()
        assert "weather" in summary
        assert "news" in summary

    def test_budget_limit(self):
        registry = AgentRegistry()
        for i in range(100):
            registry.register_skill(_make_skill(f"agent-{i}", f"Agent number {i} does things"))

        summary = registry.cards_summary(budget=200)
        assert len(summary) < 400  # some overhead for the "... and N more" line
        assert "more agents" in summary

    def test_awareness_prompt_block(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Test agent"))

        block = registry.awareness_prompt_block()
        assert "## Available Agents" in block
        assert "test" in block

    def test_awareness_prompt_block_empty(self):
        registry = AgentRegistry()
        assert registry.awareness_prompt_block() == ""


class TestRouting:
    def test_keyword_match(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("weather", "Query weather forecasts", tags=["outdoor"]))
        registry.register_skill(_make_skill("news", "Read latest news articles"))

        matches = registry.match("weather forecast today")
        assert len(matches) >= 1
        assert matches[0].card.name == "weather"

    def test_name_bonus(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("weather", "Some description"))
        registry.register_skill(_make_skill("forecast", "Weather prediction service"))

        # "weather" appears in both descriptions but only as a name for the first
        matches = registry.match("check weather")
        assert matches[0].card.name == "weather"

    def test_no_matches(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("weather", "Query weather"))

        matches = registry.match("cook a recipe")
        assert len(matches) == 0

    def test_empty_query(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Test"))

        assert registry.match("") == []

    def test_top_k_limit(self):
        registry = AgentRegistry()
        for i in range(10):
            registry.register_skill(_make_skill(f"agent-{i}", f"Agent that handles queries about topic {i}"))

        matches = registry.match("queries topic", top_k=3)
        assert len(matches) <= 3


class TestAgentStats:
    def test_initial_state(self):
        stats = AgentStats()
        assert stats.total_calls == 0
        assert stats.success_rate == 0.0
        assert stats.avg_latency_ms == 0.0

    def test_record_success(self):
        stats = AgentStats()
        stats.record_success(100.0)
        stats.record_success(200.0)

        assert stats.total_calls == 2
        assert stats.success_count == 2
        assert stats.success_rate == 1.0
        assert stats.avg_latency_ms == 150.0

    def test_record_failure(self):
        stats = AgentStats()
        stats.record_success(100.0)
        stats.record_failure("timeout", 5000.0)

        assert stats.total_calls == 2
        assert stats.success_rate == 0.5
        assert stats.last_error == "timeout"

    def test_stats_attached_to_agent(self):
        registry = AgentRegistry()
        registry.register_skill(_make_skill("test", "Test"))

        agent = registry.get("test")
        agent.stats.record_success(50.0)

        assert registry.get("test").stats.total_calls == 1
