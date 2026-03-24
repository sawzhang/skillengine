"""
Unified command registry for slash commands.

Aggregates commands from built-in handlers, extensions, skills, and prompt templates
into a single dispatch point.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skillengine.extensions.models import CommandInfo

if TYPE_CHECKING:
    from skillengine.engine import SkillsEngine
    from skillengine.extensions.manager import ExtensionManager
    from skillengine.models import Skill
    from skillengine.prompts import PromptTemplate, PromptTemplateLoader


@dataclass
class CommandResult:
    """Result of executing a command."""

    output: str = ""
    error: str = ""
    handled: bool = True  # False means pass content through to LLM
    content: str = ""  # Content to pass to LLM when handled=False


class CommandRegistry:
    """
    Unified command registry and dispatcher.

    Collects commands from:
    - Built-in commands (/quit, /clear, /skills, /help, /reload)
    - Extension-registered commands
    - User-invocable skills
    - Prompt templates
    """

    def __init__(self, engine: SkillsEngine) -> None:
        self.engine = engine
        self._commands: dict[str, CommandInfo] = {}
        self._should_quit = False

        # Register built-in commands
        self._register_builtins()

    @property
    def should_quit(self) -> bool:
        """Check if the quit command has been invoked."""
        return self._should_quit

    def _register_builtins(self) -> None:
        """Register built-in REPL commands."""
        self.register("/quit", self._cmd_quit, "Exit the REPL", source="builtin", usage="/quit")
        self.register("/exit", self._cmd_quit, "Exit the REPL", source="builtin", usage="/exit")
        self.register("/q", self._cmd_quit, "Exit the REPL", source="builtin", usage="/q")
        self.register(
            "/clear",
            self._cmd_clear,
            "Clear conversation history",
            source="builtin",
            usage="/clear",
        )
        self.register(
            "/skills",
            self._cmd_skills,
            "List available skills",
            source="builtin",
            usage="/skills",
        )
        self.register(
            "/help",
            self._cmd_help,
            "List all available commands",
            source="builtin",
            usage="/help",
        )
        self.register(
            "/reload",
            self._cmd_reload,
            "Reload skills and extensions",
            source="builtin",
            usage="/reload",
        )

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        source: str = "builtin",
        usage: str = "",
        extension_name: str = "",
    ) -> None:
        """Register a command."""
        if not name.startswith("/"):
            name = f"/{name}"
        if name in self._commands:
            # Later registration wins (with warning potential)
            pass
        self._commands[name] = CommandInfo(
            name=name,
            description=description,
            handler=handler,
            source=source,
            extension_name=extension_name,
            usage=usage or name,
        )

    def unregister(self, name: str) -> None:
        """Remove a command."""
        if not name.startswith("/"):
            name = f"/{name}"
        self._commands.pop(name, None)

    def sync_from_extensions(self, ext_manager: ExtensionManager) -> None:
        """Pull commands from the extension manager."""
        for cmd in ext_manager.get_commands():
            name = cmd.name if cmd.name.startswith("/") else f"/{cmd.name}"
            self._commands[name] = CommandInfo(
                name=name,
                description=cmd.description,
                handler=cmd.handler,
                source="extension",
                extension_name=cmd.extension_name,
                usage=cmd.usage or name,
            )

    def sync_from_skills(self, skills: list[Skill]) -> None:
        """Register user-invocable skills as commands.

        Skills with actions get a handler that routes to direct execution
        when the first arg matches an action name. Otherwise falls through to LLM.
        """
        for skill in skills:
            if not skill.metadata.invocation.user_invocable:
                continue
            name = f"/{skill.name}"
            if name in self._commands:
                continue  # don't override existing commands

            if skill.has_actions:
                # Skill with actions: try direct execution first
                self._commands[name] = CommandInfo(
                    name=name,
                    description=skill.description,
                    handler=self._make_action_handler(skill),
                    source="skill",
                    usage=self._build_action_usage(skill),
                )
            else:
                # Skill without actions: pass through to LLM
                self._commands[name] = CommandInfo(
                    name=name,
                    description=skill.description,
                    handler=self._make_llm_handler(skill),
                    source="skill",
                    usage=f"/{skill.name} [args]",
                )

    def _make_llm_handler(self, skill: Skill) -> Callable[[str], CommandResult]:
        """Create a handler that passes skill content to LLM."""
        s = skill

        def handler(args: str) -> CommandResult:
            return CommandResult(
                handled=False,
                content=(
                    f"[User invoked skill: /{s.name}]\n\n"
                    f'<skill-content name="{s.name}">\n'
                    f"{s.content}\n"
                    f"</skill-content>\n\n"
                    f"User input: /{s.name} {args}".strip()
                ),
            )

        return handler

    def _make_action_handler(self, skill: Skill) -> Callable[..., Any]:
        """Create a handler that routes to direct action execution or LLM fallback."""
        s = skill
        engine = self.engine

        async def handler(args: str) -> CommandResult:
            parts = args.split() if args else []

            # No args or first arg is not an action → show available actions
            if not parts or parts[0] not in s.actions:
                # If there are args but no matching action, fall through to LLM
                if parts:
                    return CommandResult(
                        handled=False,
                        content=(
                            f"[User invoked skill: /{s.name}]\n\n"
                            f'<skill-content name="{s.name}">\n'
                            f"{s.content}\n"
                            f"</skill-content>\n\n"
                            f"User input: /{s.name} {args}".strip()
                        ),
                    )
                # No args → list actions
                lines = [f"/{s.name} actions:"]
                for action in s.actions.values():
                    lines.append(f"  {action.name:<20} {action.description}")
                lines.append(f"\nUsage: /{s.name} <action> [args...]")
                lines.append(f"Or ask freely: /{s.name} <your request> (uses LLM)")
                return CommandResult(output="\n".join(lines))

            # Direct execution path — $0, no LLM
            action_name = parts[0]
            action_args = parts[1:]
            result = await engine.execute_action(s.name, action_name, action_args)

            if result.success:
                return CommandResult(output=result.output or "(no output)")
            return CommandResult(
                error=f"Action failed (exit {result.exit_code}): {result.error or result.output}"
            )

        return handler

    @staticmethod
    def _build_action_usage(skill: Skill) -> str:
        """Build usage string showing available actions."""
        actions = ", ".join(skill.actions.keys())
        return f"/{skill.name} <{actions}> [args...]"

    def sync_from_prompts(
        self, templates: list[PromptTemplate], loader: PromptTemplateLoader
    ) -> None:
        """Register prompt templates as commands that pass through to LLM."""
        for template in templates:
            name = f"/{template.name}"
            if name in self._commands:
                continue

            def make_handler(
                t: PromptTemplate, ld: PromptTemplateLoader
            ) -> Callable[[str], CommandResult]:
                def handler(args: str) -> CommandResult:
                    substituted = ld.substitute(t, args)
                    return CommandResult(handled=False, content=substituted)

                return handler

            self._commands[name] = CommandInfo(
                name=name,
                description=template.description,
                handler=make_handler(template, loader),
                source="prompt",
                usage=f"/{template.name} [args]",
            )

    def get(self, name: str) -> CommandInfo | None:
        """Get a command by name."""
        if not name.startswith("/"):
            name = f"/{name}"
        return self._commands.get(name)

    def list_commands(self) -> list[CommandInfo]:
        """List all registered commands, sorted by name."""
        return sorted(self._commands.values(), key=lambda c: c.name)

    async def dispatch(self, name: str, args: str = "") -> CommandResult:
        """
        Dispatch a command by name.

        Handles both sync and async handlers.
        """
        cmd = self.get(name)
        if cmd is None or cmd.handler is None:
            # Unknown command - let caller decide (e.g. pass to LLM)
            return CommandResult(handled=False)

        try:
            result = cmd.handler(args)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, CommandResult):
                return result
            return CommandResult(output=str(result) if result else "")
        except Exception as e:
            return CommandResult(error=str(e))

    def get_completions(self, prefix: str) -> list[str]:
        """Get command names matching a prefix."""
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return sorted(name for name in self._commands if name.startswith(prefix))

    # Built-in command handlers

    def _cmd_quit(self, args: str = "") -> CommandResult:
        self._should_quit = True
        return CommandResult(output="Goodbye!")

    def _cmd_clear(self, args: str = "") -> CommandResult:
        return CommandResult(output="Conversation cleared.")

    def _cmd_skills(self, args: str = "") -> CommandResult:
        skills = self.engine.filter_skills()
        if not skills:
            return CommandResult(output="No skills available.")
        lines = ["Available skills:"]
        for skill in skills:
            emoji = skill.metadata.emoji or "🔧"
            invocable = "+" if skill.metadata.invocation.user_invocable else " "
            lines.append(f"  {emoji} {skill.name} [{invocable}] - {skill.description[:50]}...")
        return CommandResult(output="\n".join(lines))

    def _cmd_help(self, args: str = "") -> CommandResult:
        commands = self.list_commands()
        if not commands:
            return CommandResult(output="No commands available.")
        lines = ["Available commands:"]
        for cmd in commands:
            # Skip aliases for quit
            if cmd.name in ("/exit", "/q"):
                continue
            source_tag = f" [{cmd.source}]" if cmd.source != "builtin" else ""
            lines.append(f"  {cmd.name:<16} {cmd.description}{source_tag}")
        return CommandResult(output="\n".join(lines))

    def _cmd_reload(self, args: str = "") -> CommandResult:
        self.engine.invalidate_cache()
        snapshot = self.engine.get_snapshot()
        return CommandResult(output=f"Reloaded. {len(snapshot.skills)} skills available.")
