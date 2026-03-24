"""Tests for skill actions - deterministic execution without LLM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from skillengine import SkillsConfig, SkillsEngine
from skillengine.commands import CommandRegistry, CommandResult
from skillengine.models import (
    Skill,
    SkillAction,
    SkillActionParam,
    SkillInvocationPolicy,
    SkillMetadata,
    SkillSource,
)


@pytest.fixture
def skill_dir_with_actions(tmp_path: Path) -> Path:
    """Create a skill directory with actions and scripts."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create a skill with actions
    tool_dir = skills_dir / "tool"
    tool_dir.mkdir()
    scripts_dir = tool_dir / "scripts"
    scripts_dir.mkdir()

    # Write a simple script that echoes args
    (scripts_dir / "greet.py").write_text(
        dedent("""
        import sys
        name = sys.argv[1] if len(sys.argv) > 1 else "world"
        print(f"Hello, {name}!")
        """).strip()
    )

    # Write a script that outputs JSON
    (scripts_dir / "info.py").write_text(
        dedent("""
        import json
        print(json.dumps({"status": "ok", "version": "1.0"}))
        """).strip()
    )

    # Write a script that fails
    (scripts_dir / "fail.py").write_text(
        dedent("""
        import sys
        print("something went wrong", file=sys.stderr)
        sys.exit(1)
        """).strip()
    )

    # SKILL.md with actions
    (tool_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: tool
        description: "A test tool with actions"
        actions:
          greet:
            script: scripts/greet.py
            description: "Greet someone"
            params:
              name: { type: string, required: false, position: 1 }
            output: text
          info:
            script: scripts/info.py
            description: "Show info as JSON"
            output: json
          fail:
            script: scripts/fail.py
            description: "A failing action"
        ---
        # Tool Skill

        This skill has both actions and documentation.
        """).strip()
    )

    # Also create a skill WITHOUT actions (pure LLM skill)
    plain_dir = skills_dir / "plain"
    plain_dir.mkdir()
    (plain_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: plain
        description: "A plain skill with no actions"
        ---
        # Plain Skill

        Just documentation for the LLM.
        """).strip()
    )

    return skills_dir


@pytest.fixture
def engine(skill_dir_with_actions: Path) -> SkillsEngine:
    config = SkillsConfig(skill_dirs=[skill_dir_with_actions])
    return SkillsEngine(config=config)


class TestActionParsing:
    def test_actions_loaded_from_frontmatter(self, engine: SkillsEngine) -> None:
        skill = engine.get_skill("tool")
        assert skill is not None
        assert skill.has_actions
        assert "greet" in skill.actions
        assert "info" in skill.actions
        assert "fail" in skill.actions

    def test_action_details(self, engine: SkillsEngine) -> None:
        skill = engine.get_skill("tool")
        assert skill is not None
        greet = skill.get_action("greet")
        assert greet is not None
        assert greet.script == "scripts/greet.py"
        assert greet.description == "Greet someone"
        assert greet.output == "text"
        assert len(greet.params) == 1
        assert greet.params[0].name == "name"
        assert greet.params[0].position == 1

    def test_skill_without_actions(self, engine: SkillsEngine) -> None:
        skill = engine.get_skill("plain")
        assert skill is not None
        assert not skill.has_actions
        assert skill.actions == {}

    def test_get_action_nonexistent(self, engine: SkillsEngine) -> None:
        skill = engine.get_skill("tool")
        assert skill is not None
        assert skill.get_action("nonexistent") is None


class TestActionExecution:
    def test_execute_action_success(self, engine: SkillsEngine) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("tool", "greet", ["Alice"])
        )
        assert result.success
        assert "Hello, Alice!" in result.output

    def test_execute_action_no_args(self, engine: SkillsEngine) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("tool", "greet")
        )
        assert result.success
        assert "Hello, world!" in result.output

    def test_execute_action_json_output(self, engine: SkillsEngine) -> None:
        import json

        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("tool", "info")
        )
        assert result.success
        data = json.loads(result.output)
        assert data["status"] == "ok"

    def test_execute_action_failure(self, engine: SkillsEngine) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("tool", "fail")
        )
        assert not result.success
        assert result.exit_code == 1

    def test_execute_action_skill_not_found(self, engine: SkillsEngine) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("nonexistent", "greet")
        )
        assert not result.success
        assert "not found" in result.error.lower()

    def test_execute_action_action_not_found(self, engine: SkillsEngine) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            engine.execute_action("tool", "nonexistent")
        )
        assert not result.success
        assert "not found" in result.error.lower()
        assert "greet" in result.error  # shows available actions


class TestActionCommandRouting:
    def test_skill_with_actions_shows_list(self, engine: SkillsEngine) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())

        # /tool with no args → show actions
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/tool")
        )
        assert result.handled
        assert "greet" in result.output
        assert "info" in result.output

    def test_skill_action_direct_execution(self, engine: SkillsEngine) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())

        # /tool greet Alice → direct execution, no LLM
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/tool", "greet Alice")
        )
        assert result.handled
        assert "Hello, Alice!" in result.output

    def test_skill_action_no_match_falls_to_llm(
        self, engine: SkillsEngine
    ) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())

        # /tool help me with something → not an action, fall to LLM
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/tool", "help me with something")
        )
        assert not result.handled
        assert result.content  # has LLM passthrough content
        assert "Tool Skill" in result.content

    def test_plain_skill_always_goes_to_llm(self, engine: SkillsEngine) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())

        # /plain anything → always LLM
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/plain", "do something")
        )
        assert not result.handled
        assert "Plain Skill" in result.content

    def test_action_failure_returns_error(self, engine: SkillsEngine) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())

        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/tool", "fail")
        )
        assert result.error
        assert "failed" in result.error.lower()

    def test_usage_shows_actions(self, engine: SkillsEngine) -> None:
        registry = CommandRegistry(engine)
        registry.sync_from_skills(engine.filter_skills())
        cmd = registry.get("/tool")
        assert cmd is not None
        assert "greet" in cmd.usage
        assert "info" in cmd.usage


class TestActionModel:
    def test_skill_action_defaults(self) -> None:
        action = SkillAction(name="test", script="test.py")
        assert action.output == "text"
        assert action.params == []
        assert action.description == ""

    def test_skill_action_param_defaults(self) -> None:
        param = SkillActionParam(name="file")
        assert param.type == "string"
        assert not param.required
        assert param.position is None

    def test_skill_has_actions(self) -> None:
        skill = Skill(
            name="test",
            description="test",
            content="test",
            file_path=Path("/tmp/test"),
            base_dir=Path("/tmp"),
            actions={"greet": SkillAction(name="greet", script="greet.py")},
        )
        assert skill.has_actions
        assert skill.get_action("greet") is not None
        assert skill.get_action("missing") is None
