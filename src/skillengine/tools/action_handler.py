"""Skill action tool handler — dispatches 'skill-name:action-name' tools.

Handles deterministic skill actions (CLI scripts with typed parameters).
Migrated from AgentRunner._execute_tool() skill action branch and
AgentRunner._build_action_args().
"""

from __future__ import annotations

import logging
from typing import Any

from skillengine.tools.dispatcher import OutputCallback, ToolContext

logger = logging.getLogger(__name__)


class SkillActionHandler:
    """Handles dispatch of skill action tools (format: 'skill-name:action-name').

    Used as the pattern_handler on ToolDispatcher for ':' separated names.
    """

    def __init__(self, agent_runner: Any) -> None:
        """Initialize with a reference to the AgentRunner.

        Needs agent_runner for get_skill() to look up skills by name.
        """
        self._runner = agent_runner

    async def __call__(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
        on_output: OutputCallback | None = None,
    ) -> str:
        """Handle a 'skill-name:action-name' tool call."""
        if ":" not in name:
            return f"Error: Invalid action tool name '{name}' (expected 'skill:action')"

        skill_name, action_name = name.split(":", 1)
        skill = self._runner.get_skill(skill_name)
        if not skill:
            return f"Error: Skill '{skill_name}' not found"

        action = skill.get_action(action_name)
        if not action:
            return f"Error: Action '{action_name}' not found in skill '{skill_name}'"

        cli_args = self._build_action_args(action, args)
        with ctx.engine.env_context():
            result = await ctx.engine.execute_action(
                skill_name,
                action_name,
                args=cli_args,
                timeout=ctx.engine.config.default_timeout_seconds,
            )
        if result.success:
            output = result.output or "(no output)"
            return output
        return f"Error (exit {result.exit_code}): {result.error}"

    @staticmethod
    def _build_action_args(
        action: Any,  # SkillAction
        named_args: dict[str, Any],
    ) -> list[str]:
        """Convert named LLM arguments into positional CLI arguments.

        Uses the ``position`` field on each ``SkillActionParam`` to place
        values in the right order. Params without a position are appended
        at the end.
        """
        positional: list[tuple[int, str]] = []
        extra: list[str] = []

        for param in action.params:
            value = named_args.get(param.name)
            if value is None:
                if param.default is not None:
                    value = param.default
                else:
                    continue
            value_str = str(value)
            if param.position is not None:
                positional.append((param.position, value_str))
            else:
                extra.append(value_str)

        positional.sort(key=lambda t: t[0])
        return [v for _, v in positional] + extra
