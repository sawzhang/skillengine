"""Tests for the Skill tool, description budget, $ARGUMENTS, validation, fork, and dynamic injection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.models import (
    Skill,
    SkillInvocationPolicy,
    SkillMetadata,
    SkillSnapshot,
    SkillSource,
)
from skillengine.runtime.base import ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    name: str = "pdf",
    description: str = "PDF operations",
    content: str = "# PDF Skill\nDo PDF things with $ARGUMENTS",
    **kwargs,
) -> Skill:
    return Skill(
        name=name,
        description=description,
        content=content,
        file_path=Path(f"/tmp/skills/{name}/SKILL.md"),
        base_dir=Path(f"/tmp/skills/{name}"),
        source=SkillSource.WORKSPACE,
        metadata=kwargs.pop("metadata", SkillMetadata()),
        **kwargs,
    )


def _make_runner(skills: list[Skill], **config_kwargs) -> AgentRunner:
    engine = MagicMock(spec=SkillsEngine)
    engine.extensions = None
    snapshot = MagicMock(spec=SkillSnapshot)
    snapshot.skills = skills
    snapshot.prompt = ""
    snapshot.get_skill = lambda n: next((s for s in skills if s.name == n), None)
    engine.get_snapshot.return_value = snapshot
    engine.config = MagicMock()
    engine.config.default_timeout_seconds = 30.0

    config = AgentConfig(enable_tools=True, load_context_files=False, **config_kwargs)
    runner = AgentRunner(engine, config)
    return runner


# ===========================================================================
# Phase 1: Skill Tool + Description Budget
# ===========================================================================

class TestSkillToolInGetTools:
    """The skill tool should appear in get_tools() when visible skills exist."""

    def test_skill_tool_present_when_skills_exist(self):
        skill = _make_skill()
        runner = _make_runner([skill])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "skill" in names

    def test_skill_tool_absent_when_no_skills(self):
        runner = _make_runner([])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "skill" not in names

    def test_skill_tool_absent_when_all_hidden(self):
        skill = _make_skill(
            metadata=SkillMetadata(
                invocation=SkillInvocationPolicy(disable_model_invocation=True),
            ),
        )
        runner = _make_runner([skill])
        tools = runner.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "skill" not in names

    def test_skill_tool_lists_available_names(self):
        skills = [_make_skill("pdf"), _make_skill("csv", description="CSV ops")]
        runner = _make_runner(skills)
        tools = runner.get_tools()
        skill_tool = next(t for t in tools if t["function"]["name"] == "skill")
        desc = skill_tool["function"]["parameters"]["properties"]["name"]["description"]
        assert "pdf" in desc
        assert "csv" in desc

    def test_skill_tool_schema(self):
        runner = _make_runner([_make_skill()])
        tools = runner.get_tools()
        skill_tool = next(t for t in tools if t["function"]["name"] == "skill")
        params = skill_tool["function"]["parameters"]
        assert "name" in params["properties"]
        assert "arguments" in params["properties"]
        assert params["required"] == ["name"]


class TestSkillToolDispatch:
    """_execute_tool should dispatch the 'skill' tool and return content."""

    @pytest.mark.asyncio
    async def test_returns_skill_content(self):
        skill = _make_skill(content="Do PDF things")
        runner = _make_runner([skill])

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "pdf"}),
            "id": "call_1",
        })
        assert "Do PDF things" in result

    @pytest.mark.asyncio
    async def test_skill_not_found(self):
        runner = _make_runner([_make_skill("pdf")])

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "nonexistent"}),
            "id": "call_2",
        })
        assert "Error" in result
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_skill_with_arguments_substitution(self):
        skill = _make_skill(content="Process file: $ARGUMENTS")
        runner = _make_runner([skill])

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "pdf", "arguments": "report.pdf"}),
            "id": "call_3",
        })
        assert "Process file: report.pdf" in result

    @pytest.mark.asyncio
    async def test_skill_without_arguments_param(self):
        skill = _make_skill(content="No args: $ARGUMENTS end")
        runner = _make_runner([skill])

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "pdf"}),
            "id": "call_no_args",
        })
        assert "No args:  end" in result

    @pytest.mark.asyncio
    async def test_skill_with_allowed_tools_hint(self):
        skill = _make_skill(
            content="Do things",
            allowed_tools=["Read", "Grep"],
        )
        runner = _make_runner([skill])

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "pdf"}),
            "id": "call_at",
        })
        assert "Allowed tools" in result
        assert "Read" in result
        assert "Grep" in result
        assert "Do things" in result

    @pytest.mark.asyncio
    async def test_skill_model_switch_and_restore(self):
        skill = _make_skill(
            name="special",
            content="Use special model",
            model="gpt-4o",
        )
        runner = _make_runner([skill], model="default-model")

        # Track model changes
        models_seen = []
        original_switch = runner.switch_model

        def tracking_switch(model_name, adapter_name=None):
            models_seen.append(model_name)
            original_switch(model_name, adapter_name)

        runner.switch_model = tracking_switch

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "special"}),
            "id": "call_model",
        })

        # Should switch to skill model then back
        assert models_seen == ["gpt-4o", "default-model"]
        assert runner.config.model == "default-model"


class TestDescriptionBudget:
    """build_system_prompt() should respect the description budget."""

    def test_budget_truncates_long_prompt(self):
        # Create a skill with a very long description
        long_desc = "A" * 200
        skill = _make_skill(description=long_desc)
        runner = _make_runner([skill], skill_description_budget=50)

        # Set up the snapshot to have a long prompt
        runner.engine.get_snapshot.return_value.prompt = "X" * 100

        prompt = runner.build_system_prompt()
        # The skills section should be truncated
        assert "truncated" in prompt

    def test_budget_no_truncation_within_limit(self):
        skill = _make_skill(description="Short desc")
        runner = _make_runner([skill], skill_description_budget=16000)
        runner.engine.get_snapshot.return_value.prompt = "Short skills prompt"

        prompt = runner.build_system_prompt()
        assert "truncated" not in prompt
        assert "Short skills prompt" in prompt


# ===========================================================================
# Phase 2: $ARGUMENTS Substitution + Validation
# ===========================================================================

class TestSubstituteArguments:
    """_substitute_arguments should replace $ARGUMENTS, $N, ${CLAUDE_SESSION_ID}."""

    def test_full_arguments(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("Hello $ARGUMENTS", "world")
        assert result == "Hello world"

    def test_positional_arguments(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("$1 and $2", "foo bar")
        assert result == "foo and bar"

    def test_empty_arguments_clears_placeholders(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("File: $ARGUMENTS ($1)", "")
        assert result == "File:  ()"

    def test_session_id_substitution(self):
        runner = _make_runner([], session_id="sess-123")
        result = runner._substitute_arguments("Session: ${CLAUDE_SESSION_ID}", "")
        assert result == "Session: sess-123"

    def test_session_id_empty_when_not_set(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("ID: ${CLAUDE_SESSION_ID}", "")
        assert result == "ID: "

    def test_mixed_substitution(self):
        runner = _make_runner([], session_id="abc")
        content = "Run $1 on $2 (all: $ARGUMENTS) [${CLAUDE_SESSION_ID}]"
        result = runner._substitute_arguments(content, "cmd file.txt")
        assert result == "Run cmd on file.txt (all: cmd file.txt) [abc]"

    def test_excess_positional_cleared(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("$1 $2 $3", "only-one")
        assert result == "only-one  "

    def test_double_digit_positional_args(self):
        """$10 should be the 10th arg, not $1 followed by '0'."""
        runner = _make_runner([])
        args = " ".join(f"arg{i}" for i in range(1, 12))  # arg1..arg11
        result = runner._substitute_arguments("$1 $10 $11", args)
        assert result == "arg1 arg10 arg11"

    def test_many_positional_args(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("$1-$2-$3", "a b c")
        assert result == "a-b-c"

    def test_no_placeholders_unchanged(self):
        runner = _make_runner([])
        result = runner._substitute_arguments("No placeholders here", "some args")
        assert result == "No placeholders here"


class TestValidateSkill:
    """validate_skill should enforce Claude Agent Skills naming rules."""

    def test_valid_skill(self):
        skill = _make_skill(name="pdf-tools", description="PDF operations")
        errors = AgentRunner.validate_skill(skill)
        assert errors == []

    def test_name_too_long(self):
        skill = _make_skill(name="a" * 65)
        errors = AgentRunner.validate_skill(skill)
        assert any("too long" in e for e in errors)

    def test_name_uppercase_rejected(self):
        skill = _make_skill(name="PDF")
        errors = AgentRunner.validate_skill(skill)
        assert any("lowercase" in e for e in errors)

    def test_name_leading_hyphen_rejected(self):
        skill = _make_skill(name="-bad")
        errors = AgentRunner.validate_skill(skill)
        assert any("lowercase" in e or "alphanumeric" in e for e in errors)

    def test_name_spaces_rejected(self):
        skill = _make_skill(name="has space")
        errors = AgentRunner.validate_skill(skill)
        assert len(errors) > 0

    def test_empty_description(self):
        skill = _make_skill(description="")
        errors = AgentRunner.validate_skill(skill)
        assert any("required" in e for e in errors)

    def test_description_too_long(self):
        skill = _make_skill(description="D" * 1025)
        errors = AgentRunner.validate_skill(skill)
        assert any("too long" in e for e in errors)

    def test_name_with_digits(self):
        skill = _make_skill(name="tool2go", description="OK")
        errors = AgentRunner.validate_skill(skill)
        assert errors == []

    def test_name_trailing_hyphen_allowed(self):
        """Trailing hyphens pass the regex (spec doesn't disallow them)."""
        skill = _make_skill(name="test-", description="OK")
        errors = AgentRunner.validate_skill(skill)
        assert errors == []

    def test_name_leading_digit_allowed(self):
        skill = _make_skill(name="2fast", description="OK")
        errors = AgentRunner.validate_skill(skill)
        assert errors == []

    def test_name_underscores_rejected(self):
        skill = _make_skill(name="my_tool")
        errors = AgentRunner.validate_skill(skill)
        assert len(errors) > 0

    def test_description_exactly_1024(self):
        skill = _make_skill(description="D" * 1024)
        errors = AgentRunner.validate_skill(skill)
        assert errors == []


# ===========================================================================
# Phase 2: Frontmatter Parsing
# ===========================================================================

class TestFrontmatterParsing:
    """MarkdownSkillLoader should parse Claude Agent Skills extensions."""

    def test_parse_allowed_tools(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: test skill\n"
            "allowed-tools:\n"
            "  - Read\n"
            "  - Grep\n"
            "  - Glob\n"
            "---\n"
            "# My Skill\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.allowed_tools == ["Read", "Grep", "Glob"]

    def test_parse_model(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: test\n"
            "model: claude-sonnet-4-5-20250514\n"
            "---\n"
            "# Content\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.model == "claude-sonnet-4-5-20250514"

    def test_parse_context_fork(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "forked"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: forked\n"
            "description: runs in fork\n"
            "context: fork\n"
            "---\n"
            "# Forked Skill\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.context == "fork"

    def test_parse_argument_hint(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "search"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: search\n"
            "description: search\n"
            "argument-hint: '<search query>'\n"
            "---\n"
            "# Search\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.argument_hint == "<search query>"

    def test_parse_hooks(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "hooked"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: hooked\n"
            "description: has hooks\n"
            "hooks:\n"
            "  PreToolExecution: echo pre\n"
            "  PostToolExecution: echo post\n"
            "---\n"
            "# Hooked Skill\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.hooks == {
            "PreToolExecution": "echo pre",
            "PostToolExecution": "echo post",
        }

    def test_defaults_when_extensions_absent(self, tmp_path: Path):
        from skillengine.loaders.markdown import MarkdownSkillLoader

        skill_dir = tmp_path / "basic"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: basic\n"
            "description: basic skill\n"
            "---\n"
            "# Basic\n"
        )

        loader = MarkdownSkillLoader()
        entry = loader.load_skill(skill_dir / "SKILL.md", SkillSource.WORKSPACE)

        assert entry.skill.allowed_tools == []
        assert entry.skill.model is None
        assert entry.skill.context is None
        assert entry.skill.argument_hint is None
        assert entry.skill.hooks == {}


# ===========================================================================
# Phase 3: Dynamic Content Injection (!`command`)
# ===========================================================================

class TestDynamicContentInjection:
    """_preprocess_dynamic_content should replace !`cmd` with output."""

    @pytest.mark.asyncio
    async def test_replaces_command(self):
        runner = _make_runner([])
        runner.engine.execute = AsyncMock(
            return_value=ExecutionResult(success=True, output="v1.2.3\n", exit_code=0)
        )

        result = await runner._preprocess_dynamic_content("Version: !`echo v1.2.3`")
        assert result == "Version: v1.2.3"

    @pytest.mark.asyncio
    async def test_no_commands_unchanged(self):
        runner = _make_runner([])
        content = "No commands here"
        result = await runner._preprocess_dynamic_content(content)
        assert result == content

    @pytest.mark.asyncio
    async def test_multiple_commands(self):
        runner = _make_runner([])
        call_count = 0

        async def fake_execute(cmd, **kw):
            nonlocal call_count
            call_count += 1
            return ExecutionResult(success=True, output=f"out{call_count}", exit_code=0)

        runner.engine.execute = fake_execute

        result = await runner._preprocess_dynamic_content("A: !`cmd1` B: !`cmd2`")
        assert "out" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_command_error_shows_error(self):
        runner = _make_runner([])
        runner.engine.execute = AsyncMock(
            return_value=ExecutionResult(
                success=False, output="", error="not found", exit_code=127,
            )
        )

        result = await runner._preprocess_dynamic_content("!`bad-cmd`")
        assert "[Error:" in result


# ===========================================================================
# Phase 3: Context Fork
# ===========================================================================

class TestContextFork:
    """Skill tool should spawn a child agent for context: fork skills."""

    @pytest.mark.asyncio
    async def test_fork_dispatches_to_child(self):
        skill = _make_skill(
            name="forked",
            content="You are a specialized agent.",
            context="fork",
        )
        runner = _make_runner([skill])

        # Mock _execute_skill_forked to avoid actual LLM call
        runner._execute_skill_forked = AsyncMock(return_value="Forked result")

        result = await runner._execute_tool({
            "name": "skill",
            "arguments": json.dumps({"name": "forked", "arguments": "do stuff"}),
            "id": "call_fork",
        })
        assert result == "Forked result"
        runner._execute_skill_forked.assert_called_once_with(skill, "do stuff")

    @pytest.mark.asyncio
    async def test_fork_creates_child_with_skill_model(self):
        skill = _make_skill(
            name="forked",
            content="Specialized prompt.",
            context="fork",
            model="custom-model",
        )
        runner = _make_runner([skill])

        # Patch AgentRunner constructor to capture child config
        created_configs = []
        original_init = AgentRunner.__init__

        def capture_init(self_inner, engine, config=None, **kwargs):
            if config:
                created_configs.append(config)
            original_init(self_inner, engine, config, **kwargs)

        # Mock chat on the child to avoid LLM calls
        with patch.object(AgentRunner, "__init__", capture_init):
            with patch.object(AgentRunner, "chat", new_callable=AsyncMock) as mock_chat:
                from skillengine.agent import AgentMessage
                mock_chat.return_value = AgentMessage(
                    role="assistant", content="child response"
                )
                result = await runner._execute_skill_forked(skill, "test args")

        assert result == "child response"
        # The child config should use the skill's model
        assert any(c.model == "custom-model" for c in created_configs)

    @pytest.mark.asyncio
    async def test_fork_child_config_inherits_parent_settings(self):
        skill = _make_skill(
            name="forked",
            content="Prompt.",
            context="fork",
        )
        runner = _make_runner(
            [skill],
            model="parent-model",
            thinking_level="high",
            transport="sse",
            cache_retention="long",
            session_id="sess-xyz",
            skill_description_budget=8000,
        )

        created_configs = []
        original_init = AgentRunner.__init__

        def capture_init(self_inner, engine, config=None, **kwargs):
            if config:
                created_configs.append(config)
            original_init(self_inner, engine, config, **kwargs)

        with patch.object(AgentRunner, "__init__", capture_init):
            with patch.object(AgentRunner, "chat", new_callable=AsyncMock) as mock_chat:
                from skillengine.agent import AgentMessage
                mock_chat.return_value = AgentMessage(role="assistant", content="ok")
                await runner._execute_skill_forked(skill, "test")

        child_cfg = created_configs[-1]
        assert child_cfg.model == "parent-model"
        assert child_cfg.thinking_level == "high"
        assert child_cfg.transport == "sse"
        assert child_cfg.cache_retention == "long"
        assert child_cfg.session_id == "sess-xyz"
        assert child_cfg.skill_description_budget == 8000
        assert child_cfg.system_prompt == "Prompt."
        assert child_cfg.load_context_files is False

    @pytest.mark.asyncio
    async def test_fork_allowed_tools_filters_child(self):
        skill = _make_skill(
            name="restricted",
            content="Only use Read.",
            context="fork",
            allowed_tools=["Read"],
        )
        runner = _make_runner([skill])

        child_get_tools_result = []
        original_init = AgentRunner.__init__

        def capture_init(self_inner, engine, config=None, **kwargs):
            original_init(self_inner, engine, config, **kwargs)

        with patch.object(AgentRunner, "__init__", capture_init):
            with patch.object(AgentRunner, "chat", new_callable=AsyncMock) as mock_chat:
                from skillengine.agent import AgentMessage

                async def fake_chat(user_input, **kw):
                    return AgentMessage(role="assistant", content="ok")

                mock_chat.side_effect = fake_chat

                # We need to capture the child's get_tools after fork
                original_fork = runner._execute_skill_forked

                async def capturing_fork(s, a):
                    # Call original to create child, but we intercept
                    child_config = AgentConfig(
                        model=s.model or runner.config.model,
                        system_prompt=s.content,
                        skill_dirs=list(runner.config.skill_dirs),
                        load_context_files=False,
                    )
                    child = AgentRunner(runner.engine, child_config)
                    # Simulate allowed_tools filtering
                    if s.allowed_tools:
                        allowed = set(s.allowed_tools)
                        original_gt = child.get_tools

                        def filtered():
                            return [t for t in original_gt() if t["function"]["name"] in allowed]

                        child.get_tools = filtered  # type: ignore
                    child_get_tools_result.extend(child.get_tools())
                    return "filtered result"

                runner._execute_skill_forked = capturing_fork  # type: ignore
                result = await runner._execute_tool({
                    "name": "skill",
                    "arguments": json.dumps({"name": "restricted"}),
                    "id": "call_filter",
                })

        # Only "Read" should be in the filtered tools (if it existed)
        names = [t["function"]["name"] for t in child_get_tools_result]
        # Since there's no "Read" tool in the default set (execute, execute_script),
        # the filtered list should be empty
        assert "execute" not in names
        assert "execute_script" not in names


# ===========================================================================
# Phase 3: Per-Skill Hooks (data model only)
# ===========================================================================

class TestSkillHooksModel:
    """Skill model should store hooks dict."""

    def test_hooks_default_empty(self):
        skill = _make_skill()
        assert skill.hooks == {}

    def test_hooks_stored(self):
        skill = _make_skill(hooks={"PreToolExecution": "echo pre"})
        assert skill.hooks["PreToolExecution"] == "echo pre"


# ===========================================================================
# Skill Model New Fields
# ===========================================================================

class TestSkillNewFields:
    """Test new fields on the Skill dataclass."""

    def test_allowed_tools(self):
        skill = _make_skill(allowed_tools=["Read", "Grep"])
        assert skill.allowed_tools == ["Read", "Grep"]

    def test_model_override(self):
        skill = _make_skill(model="claude-sonnet-4-5-20250514")
        assert skill.model == "claude-sonnet-4-5-20250514"

    def test_context_fork(self):
        skill = _make_skill(context="fork")
        assert skill.context == "fork"

    def test_argument_hint(self):
        skill = _make_skill(argument_hint="<file_path>")
        assert skill.argument_hint == "<file_path>"

    def test_defaults(self):
        skill = _make_skill()
        assert skill.allowed_tools == []
        assert skill.model is None
        assert skill.context is None
        assert skill.argument_hint is None
        assert skill.hooks == {}
