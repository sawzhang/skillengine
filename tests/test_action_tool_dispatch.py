"""Tests for skill action → tool generation and dispatch."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.models import (
    Skill,
    SkillAction,
    SkillActionParam,
    SkillMetadata,
    SkillSnapshot,
    SkillSource,
)
from skillengine.runtime.base import ExecutionResult


def _make_skill_with_actions(name: str = "pdf", actions: dict | None = None) -> Skill:
    """Create a skill with action definitions."""
    if actions is None:
        actions = {
            "extract-fields": SkillAction(
                name="extract-fields",
                script="scripts/extract_form_field_info.py",
                description="Extract form field info from a PDF",
                params=[
                    SkillActionParam(
                        name="input_file", type="file", required=True, position=1,
                        description="PDF file to extract from",
                    ),
                ],
                output="json",
            ),
            "fill-form": SkillAction(
                name="fill-form",
                script="scripts/fill_fillable_fields.py",
                description="Fill PDF form fields from JSON data",
                params=[
                    SkillActionParam(
                        name="input_file", type="file", required=True, position=1,
                        description="Input PDF file",
                    ),
                    SkillActionParam(
                        name="fields_json", type="file", required=True, position=2,
                        description="JSON file with field values",
                    ),
                    SkillActionParam(
                        name="output_file", type="file", required=True, position=3,
                        description="Output PDF path",
                    ),
                ],
                output="file",
            ),
        }
    return Skill(
        name=name,
        description="PDF operations",
        content="# PDF Skill",
        file_path=Path("/tmp/skills/pdf/SKILL.md"),
        base_dir=Path("/tmp/skills/pdf"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(),
        actions=actions,
    )


def _make_runner_with_skills(skills: list[Skill]) -> AgentRunner:
    """Create an AgentRunner with mocked engine returning given skills."""
    engine = MagicMock(spec=SkillsEngine)
    engine.extensions = None
    snapshot = MagicMock(spec=SkillSnapshot)
    snapshot.skills = skills
    snapshot.prompt = ""
    snapshot.get_skill = lambda n: next((s for s in skills if s.name == n), None)
    engine.get_snapshot.return_value = snapshot
    engine.config = MagicMock()
    engine.config.default_timeout_seconds = 30.0

    config = AgentConfig(enable_tools=True)
    runner = AgentRunner(engine, config)
    return runner


class TestGetToolsGeneratesActionTools:
    def test_no_actions_no_extra_tools(self):
        skill = Skill(
            name="simple",
            description="Simple skill",
            content="# Simple",
            file_path=Path("/tmp/skills/simple/SKILL.md"),
            base_dir=Path("/tmp/skills/simple"),
            source=SkillSource.WORKSPACE,
        )
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        # 4 builtin (execute, execute_script, write, read) + skill tool (on-demand loading)
        assert names == ["execute", "execute_script", "write", "read", "skill"]

    def test_skill_actions_generate_tools(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "pdf:extract-fields" in names
        assert "pdf:fill-form" in names
        assert len(tools) == 7  # 4 builtin + skill tool + 2 actions

    def test_action_tool_schema_correct(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()

        extract_tool = next(
            t for t in tools if t["function"]["name"] == "pdf:extract-fields"
        )
        func = extract_tool["function"]
        assert func["description"] == "Extract form field info from a PDF"
        params = func["parameters"]
        assert params["type"] == "object"
        assert "input_file" in params["properties"]
        assert params["properties"]["input_file"]["type"] == "string"
        assert "(file path)" in params["properties"]["input_file"]["description"]
        assert params["required"] == ["input_file"]

    def test_multi_param_action_schema(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()

        fill_tool = next(
            t for t in tools if t["function"]["name"] == "pdf:fill-form"
        )
        params = fill_tool["function"]["parameters"]
        assert set(params["properties"].keys()) == {
            "input_file", "fields_json", "output_file",
        }
        assert set(params["required"]) == {
            "input_file", "fields_json", "output_file",
        }

    def test_param_types_mapped_correctly(self):
        actions = {
            "test-action": SkillAction(
                name="test-action",
                script="test.py",
                params=[
                    SkillActionParam(name="path", type="file", required=True),
                    SkillActionParam(name="count", type="number"),
                    SkillActionParam(name="verbose", type="bool"),
                    SkillActionParam(name="name", type="string"),
                    SkillActionParam(name="data", type="json"),
                ],
            ),
        }
        skill = _make_skill_with_actions("test", actions)
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()

        action_tool = next(
            t for t in tools if t["function"]["name"] == "test:test-action"
        )
        props = action_tool["function"]["parameters"]["properties"]
        assert props["path"]["type"] == "string"
        assert props["count"]["type"] == "number"
        assert props["verbose"]["type"] == "boolean"
        assert props["name"]["type"] == "string"
        assert props["data"]["type"] == "string"  # json → string

    def test_param_default_included(self):
        actions = {
            "act": SkillAction(
                name="act",
                script="s.py",
                params=[
                    SkillActionParam(
                        name="fmt", type="string", default="png",
                        description="Output format",
                    ),
                ],
            ),
        }
        skill = _make_skill_with_actions("img", actions)
        runner = _make_runner_with_skills([skill])
        tools = runner.get_tools()

        act_tool = next(t for t in tools if t["function"]["name"] == "img:act")
        assert act_tool["function"]["parameters"]["properties"]["fmt"]["default"] == "png"

    def test_multiple_skills_with_actions(self):
        pdf_skill = _make_skill_with_actions("pdf")
        pptx_actions = {
            "inventory": SkillAction(
                name="inventory",
                script="scripts/inventory.py",
                description="Extract text from slides",
                params=[
                    SkillActionParam(
                        name="input_file", type="file", required=True, position=1,
                    ),
                    SkillActionParam(
                        name="output_file", type="file", required=True, position=2,
                    ),
                ],
                output="json",
            ),
        }
        pptx_skill = _make_skill_with_actions("pptx", pptx_actions)
        runner = _make_runner_with_skills([pdf_skill, pptx_skill])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "pdf:extract-fields" in names
        assert "pdf:fill-form" in names
        assert "pptx:inventory" in names
        assert len(tools) == 8  # 4 builtin + skill tool + 2 pdf + 1 pptx


class TestBuildActionArgs:
    def test_positional_ordering(self):
        action = SkillAction(
            name="fill",
            script="fill.py",
            params=[
                SkillActionParam(name="output", position=3, required=True),
                SkillActionParam(name="input", position=1, required=True),
                SkillActionParam(name="data", position=2, required=True),
            ],
        )
        args = AgentRunner._build_action_args(
            action, {"input": "in.pdf", "data": "d.json", "output": "out.pdf"}
        )
        assert args == ["in.pdf", "d.json", "out.pdf"]

    def test_missing_optional_param_skipped(self):
        action = SkillAction(
            name="test",
            script="t.py",
            params=[
                SkillActionParam(name="file", position=1, required=True),
                SkillActionParam(name="verbose", position=2, required=False),
            ],
        )
        args = AgentRunner._build_action_args(action, {"file": "f.txt"})
        assert args == ["f.txt"]

    def test_default_value_used(self):
        action = SkillAction(
            name="test",
            script="t.py",
            params=[
                SkillActionParam(name="file", position=1, required=True),
                SkillActionParam(name="fmt", position=2, default="png"),
            ],
        )
        args = AgentRunner._build_action_args(action, {"file": "img.pdf"})
        assert args == ["img.pdf", "png"]

    def test_no_position_appended_at_end(self):
        action = SkillAction(
            name="test",
            script="t.py",
            params=[
                SkillActionParam(name="file", position=1, required=True),
                SkillActionParam(name="extra"),
            ],
        )
        args = AgentRunner._build_action_args(
            action, {"file": "a.txt", "extra": "val"}
        )
        assert args == ["a.txt", "val"]

    def test_empty_args(self):
        action = SkillAction(name="test", script="t.py", params=[])
        args = AgentRunner._build_action_args(action, {})
        assert args == []


class TestExecuteToolActionDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_action_tool(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])

        # Mock execute_action
        runner.engine.execute_action = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                output='{"fields": ["name", "address"]}',
                exit_code=0,
            )
        )
        runner.engine.env_context = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))

        result = await runner._execute_tool({
            "name": "pdf:extract-fields",
            "arguments": json.dumps({"input_file": "/tmp/form.pdf"}),
            "id": "call_1",
        })
        assert '{"fields"' in result
        runner.engine.execute_action.assert_called_once_with(
            "pdf",
            "extract-fields",
            args=["/tmp/form.pdf"],
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_action_tool_multi_param(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])

        runner.engine.execute_action = AsyncMock(
            return_value=ExecutionResult(success=True, output="Done", exit_code=0)
        )
        runner.engine.env_context = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))

        result = await runner._execute_tool({
            "name": "pdf:fill-form",
            "arguments": json.dumps({
                "input_file": "form.pdf",
                "fields_json": "data.json",
                "output_file": "filled.pdf",
            }),
            "id": "call_2",
        })
        assert result == "Done"
        call_args = runner.engine.execute_action.call_args
        assert call_args[1]["args"] == ["form.pdf", "data.json", "filled.pdf"]

    @pytest.mark.asyncio
    async def test_action_tool_error(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])

        runner.engine.execute_action = AsyncMock(
            return_value=ExecutionResult(
                success=False, output="", error="Script crashed", exit_code=1,
            )
        )
        runner.engine.env_context = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))

        result = await runner._execute_tool({
            "name": "pdf:extract-fields",
            "arguments": json.dumps({"input_file": "bad.pdf"}),
            "id": "call_3",
        })
        assert "Error" in result
        assert "Script crashed" in result

    @pytest.mark.asyncio
    async def test_unknown_action_falls_through(self):
        skill = _make_skill_with_actions()
        runner = _make_runner_with_skills([skill])

        result = await runner._execute_tool({
            "name": "pdf:nonexistent",
            "arguments": "{}",
            "id": "call_4",
        })
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_unknown_skill_falls_through(self):
        runner = _make_runner_with_skills([])

        result = await runner._execute_tool({
            "name": "nope:action",
            "arguments": "{}",
            "id": "call_5",
        })
        assert "Unknown tool" in result


class TestEnvContextInjection:
    @pytest.mark.asyncio
    async def test_execute_wrapped_in_env_context(self):
        runner = _make_runner_with_skills([])

        env_entered = False

        class FakeCtx:
            def __enter__(self_ctx):
                nonlocal env_entered
                env_entered = True
                return None

            def __exit__(self_ctx, *args):
                return False

        runner.engine.env_context = MagicMock(return_value=FakeCtx())
        runner.engine.execute = AsyncMock(
            return_value=ExecutionResult(success=True, output="ok", exit_code=0)
        )

        await runner._execute_tool({
            "name": "execute",
            "arguments": json.dumps({"command": "echo hi"}),
            "id": "call_6",
        })
        assert env_entered

    @pytest.mark.asyncio
    async def test_execute_script_wrapped_in_env_context(self):
        runner = _make_runner_with_skills([])

        env_entered = False

        class FakeCtx:
            def __enter__(self_ctx):
                nonlocal env_entered
                env_entered = True
                return None

            def __exit__(self_ctx, *args):
                return False

        runner.engine.env_context = MagicMock(return_value=FakeCtx())
        runner.engine.execute_script = AsyncMock(
            return_value=ExecutionResult(success=True, output="ok", exit_code=0)
        )

        await runner._execute_tool({
            "name": "execute_script",
            "arguments": json.dumps({"script": "echo hi"}),
            "id": "call_7",
        })
        assert env_entered
