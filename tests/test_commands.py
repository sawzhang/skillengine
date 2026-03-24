"""Tests for the command registry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skillengine import SkillsConfig, SkillsEngine
from skillengine.commands import CommandRegistry, CommandResult
from skillengine.extensions.manager import ExtensionManager
from skillengine.extensions.models import CommandInfo
from skillengine.models import (
    Skill,
    SkillInvocationPolicy,
    SkillMetadata,
    SkillSource,
)
from skillengine.prompts import PromptTemplate, PromptTemplateLoader


@pytest.fixture
def engine(skill_dir: Path) -> SkillsEngine:
    config = SkillsConfig(skill_dirs=[skill_dir])
    return SkillsEngine(config=config)


@pytest.fixture
def registry(engine: SkillsEngine) -> CommandRegistry:
    return CommandRegistry(engine)


class TestBuiltinCommands:
    def test_quit_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/quit")
        )
        assert result.handled
        assert registry.should_quit
        assert "Goodbye" in result.output

    def test_exit_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/exit")
        )
        assert registry.should_quit

    def test_q_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/q")
        )
        assert registry.should_quit

    def test_clear_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/clear")
        )
        assert result.handled
        assert "cleared" in result.output.lower()

    def test_skills_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/skills")
        )
        assert result.handled
        assert "skills" in result.output.lower()

    def test_help_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/help")
        )
        assert result.handled
        assert "/quit" in result.output
        assert "/clear" in result.output
        assert "/skills" in result.output
        assert "/help" in result.output
        assert "/reload" in result.output

    def test_reload_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/reload")
        )
        assert result.handled
        assert "Reloaded" in result.output

    def test_unknown_command(self, registry: CommandRegistry) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/nonexistent")
        )
        assert not result.handled
        assert not result.error  # no error, just unhandled (fallthrough to LLM)


class TestSyncFromSkills:
    def test_user_invocable_skills_become_commands(
        self, registry: CommandRegistry
    ) -> None:
        skill = Skill(
            name="pdf",
            description="Generate PDF",
            content="# PDF\nGenerate a PDF.",
            file_path=Path("/tmp/skills/pdf/SKILL.md"),
            base_dir=Path("/tmp/skills/pdf"),
            source=SkillSource.WORKSPACE,
            metadata=SkillMetadata(
                invocation=SkillInvocationPolicy(user_invocable=True)
            ),
        )
        registry.sync_from_skills([skill])
        cmd = registry.get("/pdf")
        assert cmd is not None
        assert cmd.source == "skill"

    def test_non_invocable_skills_excluded(self, registry: CommandRegistry) -> None:
        skill = Skill(
            name="internal",
            description="Internal tool",
            content="# Internal",
            file_path=Path("/tmp/skills/internal/SKILL.md"),
            base_dir=Path("/tmp/skills/internal"),
            source=SkillSource.WORKSPACE,
            metadata=SkillMetadata(
                invocation=SkillInvocationPolicy(user_invocable=False)
            ),
        )
        registry.sync_from_skills([skill])
        assert registry.get("/internal") is None

    def test_skill_command_returns_unhandled(self, registry: CommandRegistry) -> None:
        skill = Skill(
            name="pdf",
            description="Generate PDF",
            content="# PDF Content",
            file_path=Path("/tmp/skills/pdf/SKILL.md"),
            base_dir=Path("/tmp/skills/pdf"),
            source=SkillSource.WORKSPACE,
            metadata=SkillMetadata(
                invocation=SkillInvocationPolicy(user_invocable=True)
            ),
        )
        registry.sync_from_skills([skill])
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/pdf", "myfile.txt")
        )
        assert not result.handled
        assert "PDF Content" in result.content
        assert "myfile.txt" in result.content


class TestSyncFromPrompts:
    def test_templates_become_commands(self, registry: CommandRegistry) -> None:
        template = PromptTemplate(
            name="review",
            content="Review this code focusing on $1.\n\n$@",
            description="Review code changes",
        )
        loader = PromptTemplateLoader()
        registry.sync_from_prompts([template], loader)
        cmd = registry.get("/review")
        assert cmd is not None
        assert cmd.source == "prompt"

    def test_prompt_command_returns_unhandled(
        self, registry: CommandRegistry
    ) -> None:
        template = PromptTemplate(
            name="review",
            content="Review this code focusing on $1.",
            description="Review code",
        )
        loader = PromptTemplateLoader()
        registry.sync_from_prompts([template], loader)
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/review", "security")
        )
        assert not result.handled
        assert "security" in result.content


class TestSyncFromExtensions:
    def test_extension_commands_synced(
        self, registry: CommandRegistry, engine: SkillsEngine
    ) -> None:
        ext_manager = ExtensionManager(engine)
        ext_manager._register_command(
            "deploy", lambda args: "deployed", "Deploy app", "ext1", "/deploy"
        )
        registry.sync_from_extensions(ext_manager)
        cmd = registry.get("/deploy")
        assert cmd is not None
        assert cmd.source == "extension"


class TestCompletions:
    def test_get_completions(self, registry: CommandRegistry) -> None:
        matches = registry.get_completions("/q")
        assert "/quit" in matches
        assert "/q" in matches

    def test_get_completions_all(self, registry: CommandRegistry) -> None:
        matches = registry.get_completions("/")
        assert len(matches) >= 5  # at least the builtins

    def test_get_completions_no_match(self, registry: CommandRegistry) -> None:
        matches = registry.get_completions("/zzz")
        assert matches == []


class TestRegisterUnregister:
    def test_register_and_get(self, registry: CommandRegistry) -> None:
        registry.register("/test", lambda args: "ok", "Test command")
        cmd = registry.get("/test")
        assert cmd is not None
        assert cmd.description == "Test command"

    def test_unregister(self, registry: CommandRegistry) -> None:
        registry.register("/test", lambda args: "ok", "Test")
        registry.unregister("/test")
        assert registry.get("/test") is None

    def test_auto_prefix_slash(self, registry: CommandRegistry) -> None:
        registry.register("test", lambda args: "ok", "Test")
        assert registry.get("/test") is not None
        assert registry.get("test") is not None


class TestAsyncHandler:
    def test_async_handler_dispatch(self, registry: CommandRegistry) -> None:
        async def async_handler(args: str) -> CommandResult:
            return CommandResult(output="async done")

        registry.register("/async-cmd", async_handler, "Async command")
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/async-cmd")
        )
        assert result.output == "async done"

    def test_handler_exception(self, registry: CommandRegistry) -> None:
        def bad_handler(args: str) -> None:
            raise ValueError("oops")

        registry.register("/bad", bad_handler, "Bad command")
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch("/bad")
        )
        assert "oops" in result.error
