"""Tests for AgentCard generation from Skills."""

from pathlib import Path

from skillkit.a2a.agent_card import AgentCard, AgentCapabilities, AgentCardSkill
from skillkit.models import Skill, SkillMetadata


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    tags: list[str] | None = None,
    version: str | None = "1.0.0",
    author: str | None = "tester",
    model: str | None = None,
    a2a_config: dict | None = None,
) -> Skill:
    """Create a Skill for testing."""
    skill = Skill(
        name=name,
        description=description,
        content="# Test\nDo the thing.",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        metadata=SkillMetadata(
            version=version,
            author=author,
            tags=tags or [],
        ),
    )
    if model:
        skill.model = model
    if a2a_config:
        skill._a2a_config = a2a_config
    return skill


class TestAgentCardFromSkill:
    def test_basic_conversion(self):
        skill = _make_skill()
        card = AgentCard.from_skill(skill)

        assert card.name == "test-skill"
        assert card.description == "A test skill"
        assert card.version == "1.0.0"
        assert card.author == "tester"
        assert len(card.skills) == 1
        assert card.skills[0].name == "test-skill"

    def test_with_base_url(self):
        skill = _make_skill()
        card = AgentCard.from_skill(skill, base_url="http://localhost:8080")

        assert card.url == "http://localhost:8080/agents/test-skill"

    def test_without_base_url(self):
        skill = _make_skill()
        card = AgentCard.from_skill(skill)

        assert card.url is None

    def test_tags_propagation(self):
        skill = _make_skill(tags=["twitter", "analysis"])
        card = AgentCard.from_skill(skill)

        assert card.tags == ["twitter", "analysis"]
        assert card.skills[0].tags == ["twitter", "analysis"]

    def test_model_propagation(self):
        skill = _make_skill(model="claude-haiku-4-5-20250514")
        card = AgentCard.from_skill(skill)

        assert card.model == "claude-haiku-4-5-20250514"

    def test_a2a_config_schemas(self):
        skill = _make_skill(
            a2a_config={
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"r": {"type": "string"}}},
            }
        )
        card = AgentCard.from_skill(skill)

        assert card.skills[0].input_schema is not None
        assert card.skills[0].input_schema["properties"]["q"]["type"] == "string"
        assert card.skills[0].output_schema is not None

    def test_a2a_config_capabilities(self):
        skill = _make_skill(a2a_config={"stateful": True, "streaming": True})
        card = AgentCard.from_skill(skill)

        assert card.capabilities.multi_turn is True
        assert card.capabilities.streaming is True

    def test_a2a_config_cost_hint(self):
        skill = _make_skill(a2a_config={"cost_hint": "high"})
        card = AgentCard.from_skill(skill)

        assert card.cost_hint == "high"

    def test_default_capabilities(self):
        skill = _make_skill()
        card = AgentCard.from_skill(skill)

        assert card.capabilities.streaming is False
        assert card.capabilities.multi_turn is False
        assert card.capabilities.push_notifications is False

    def test_default_version_when_none(self):
        skill = _make_skill(version=None)
        card = AgentCard.from_skill(skill)

        assert card.version == "1.0.0"


class TestAgentCardSerialization:
    def test_to_dict_minimal(self):
        card = AgentCard(name="test", description="Test agent")
        d = card.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test agent"
        assert d["version"] == "1.0.0"
        assert "url" not in d  # None fields excluded

    def test_to_dict_full(self):
        card = AgentCard(
            name="analyzer",
            description="Analyzes data",
            version="2.0.0",
            url="http://example.com/agents/analyzer",
            author="dev",
            tags=["data", "ml"],
            cost_hint="medium",
            model="sonnet",
            skills=[AgentCardSkill(name="analyze", description="Do analysis")],
        )
        d = card.to_dict()

        assert d["url"] == "http://example.com/agents/analyzer"
        assert d["author"] == "dev"
        assert d["tags"] == ["data", "ml"]
        assert d["cost_hint"] == "medium"
        assert d["model"] == "sonnet"
        assert len(d["skills"]) == 1

    def test_roundtrip(self):
        original = AgentCard(
            name="test",
            description="Test",
            version="1.2.0",
            author="me",
            tags=["a", "b"],
            skills=[
                AgentCardSkill(
                    name="s1",
                    description="Skill one",
                    input_schema={"type": "object"},
                )
            ],
            capabilities=AgentCapabilities(streaming=True),
        )
        d = original.to_dict()
        restored = AgentCard.from_dict(d)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.author == original.author
        assert restored.tags == original.tags
        assert len(restored.skills) == 1
        assert restored.skills[0].input_schema == {"type": "object"}
        assert restored.capabilities.streaming is True


class TestAgentCardText:
    def test_embedding_text(self):
        card = AgentCard(
            name="weather",
            description="Query weather forecasts",
            tags=["weather", "outdoor"],
            skills=[AgentCardSkill(name="weather", description="Query weather forecasts")],
        )
        text = card.to_embedding_text()

        assert "weather" in text
        assert "outdoor" in text
        assert "forecast" in text.lower()

    def test_embedding_text_no_duplicate_description(self):
        card = AgentCard(
            name="test",
            description="Same description",
            skills=[AgentCardSkill(name="test", description="Same description")],
        )
        text = card.to_embedding_text()
        # Description should appear once, not twice
        assert text.count("Same description") == 1

    def test_summary_line_with_tags(self):
        card = AgentCard(name="analyzer", description="Analyze data", tags=["ml", "data"])
        line = card.to_summary_line()

        assert "**analyzer**" in line
        assert "Analyze data" in line
        assert "[ml, data]" in line

    def test_summary_line_no_tags(self):
        card = AgentCard(name="basic", description="Basic agent")
        line = card.to_summary_line()

        assert "**basic**" in line
        assert "[" not in line
