"""Tests for Claude SDK Bridge."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from skillkit.a2a.claude_sdk_bridge import ClaudeSDKBridge
from skillkit.models import Skill, SkillMetadata


def _make_skill(
    name: str = "test-skill",
    description: str = "A test skill",
    content: str = "# Test\nDo the thing.",
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    a2a_config: dict | None = None,
) -> Skill:
    skill = Skill(
        name=name,
        description=description,
        content=content,
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        metadata=SkillMetadata(),
        allowed_tools=allowed_tools or [],
        model=model,
    )
    if a2a_config:
        skill._a2a_config = a2a_config
    return skill


def _make_engine(skills: list[Skill]) -> MagicMock:
    engine = MagicMock()
    engine.load_skills.return_value = skills

    def get_skill(name):
        for s in skills:
            if s.name == name:
                return s
        return None

    engine.get_skill = get_skill
    return engine


class TestToSdkAgentDefinition:
    def test_basic(self):
        skill = _make_skill()
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        defn = bridge.to_sdk_agent_definition("test-skill")

        assert defn["name"] == "test-skill"
        assert defn["description"] == "A test skill"
        assert defn["instructions"] == "# Test\nDo the thing."

    def test_with_tools(self):
        skill = _make_skill(allowed_tools=["Read", "Bash", "Grep"])
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        defn = bridge.to_sdk_agent_definition("test-skill")
        assert defn["tools"] == ["Read", "Bash", "Grep"]

    def test_without_tools(self):
        skill = _make_skill(allowed_tools=[])
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        defn = bridge.to_sdk_agent_definition("test-skill")
        assert "tools" not in defn

    def test_with_model(self):
        skill = _make_skill(model="claude-haiku-4-5-20250514")
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        defn = bridge.to_sdk_agent_definition("test-skill")
        assert defn["model"] == "claude-haiku-4-5-20250514"

    def test_model_override(self):
        skill = _make_skill(model="haiku")
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        defn = bridge.to_sdk_agent_definition("test-skill", model_override="opus")
        assert defn["model"] == "opus"

    def test_skill_not_found(self):
        engine = _make_engine([])
        bridge = ClaudeSDKBridge(engine)

        with pytest.raises(ValueError, match="not found"):
            bridge.to_sdk_agent_definition("nonexistent")


class TestToSdkAgents:
    def test_export_all(self):
        skills = [
            _make_skill(name="a", description="Agent A"),
            _make_skill(name="b", description="Agent B"),
        ]
        engine = _make_engine(skills)
        bridge = ClaudeSDKBridge(engine)

        agents = bridge.to_sdk_agents()
        assert len(agents) == 2
        assert "a" in agents
        assert "b" in agents

    def test_export_specific(self):
        skills = [
            _make_skill(name="a", description="Agent A"),
            _make_skill(name="b", description="Agent B"),
            _make_skill(name="c", description="Agent C"),
        ]
        engine = _make_engine(skills)
        bridge = ClaudeSDKBridge(engine)

        agents = bridge.to_sdk_agents(skill_names=["a", "c"])
        assert len(agents) == 2
        assert "a" in agents
        assert "c" in agents
        assert "b" not in agents

    def test_skip_missing(self):
        skills = [_make_skill(name="a", description="Agent A")]
        engine = _make_engine(skills)
        bridge = ClaudeSDKBridge(engine)

        agents = bridge.to_sdk_agents(skill_names=["a", "nonexistent"])
        assert len(agents) == 1
        assert "a" in agents


class TestToMcpTool:
    def test_basic(self):
        skill = _make_skill()
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        tool = bridge.to_mcp_tool("test-skill")
        assert tool["name"] == "test-skill"
        assert tool["description"] == "A test skill"
        assert "inputSchema" in tool

    def test_with_a2a_input_schema(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        skill = _make_skill(a2a_config={"input_schema": schema})
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        tool = bridge.to_mcp_tool("test-skill")
        assert tool["inputSchema"] == schema

    def test_default_schema(self):
        skill = _make_skill()
        engine = _make_engine([skill])
        bridge = ClaudeSDKBridge(engine)

        tool = bridge.to_mcp_tool("test-skill")
        assert tool["inputSchema"]["type"] == "object"
        assert "input" in tool["inputSchema"]["properties"]
