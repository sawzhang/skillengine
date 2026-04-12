"""Skill tool handler — dispatches the 'skill' meta-tool.

Handles on-demand skill loading with $ARGUMENTS substitution,
dynamic content injection (!`command`), per-skill model switching,
and context fork isolation.

Migrated from AgentRunner._execute_tool() skill branch.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from skillengine.tools.dispatcher import OutputCallback, ToolContext

logger = logging.getLogger(__name__)


class SkillToolHandler:
    """Handles dispatch of the 'skill' tool.

    Encapsulates skill-specific logic: argument substitution,
    dynamic content preprocessing, model switching, fork execution,
    and allowed-tools enforcement.
    """

    def __init__(self, agent_runner: Any) -> None:
        """Initialize with a reference to the AgentRunner.

        The agent_runner is needed for:
        - get_skill() to look up skills by name
        - skills property for listing available skills
        - switch_model() for per-skill model switching
        - config for session_id and model
        - engine for dynamic content execution
        - _execute_skill_forked() for fork context
        """
        self._runner = agent_runner

    async def __call__(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
        on_output: OutputCallback | None = None,
    ) -> str:
        """Handle a 'skill' tool call."""
        skill_name = args.get("name", "")
        arguments = args.get("arguments", "")
        skill = self._runner.get_skill(skill_name)
        if skill is None:
            available = ", ".join(s.name for s in self._runner.skills)
            return f"Error: Skill '{skill_name}' not found. Available: {available}"

        # Per-skill model switching
        previous_model: str | None = None
        if skill.model and skill.model != ctx.config.model:
            previous_model = ctx.config.model
            self._runner.switch_model(skill.model)

        try:
            content = skill.content

            # $ARGUMENTS substitution
            content = self._substitute_arguments(content, arguments, ctx)

            # Dynamic content injection (!`command`)
            content = await self._preprocess_dynamic_content(content, ctx)

            # Context fork: run in isolated subagent
            if skill.context == "fork":
                return await self._execute_skill_forked(skill, arguments, ctx)

            # Prepend allowed-tools hint if restricted
            if skill.allowed_tools:
                tools_hint = ", ".join(skill.allowed_tools)
                content = f"[Allowed tools for this skill: {tools_hint}]\n\n" + content

            return content
        finally:
            # Restore model if switched
            if previous_model is not None:
                self._runner.switch_model(previous_model)

    @staticmethod
    def _substitute_arguments(content: str, arguments: str, ctx: ToolContext) -> str:
        """Substitute $ARGUMENTS, $N, ${CLAUDE_SESSION_ID} in skill content.

        Follows Claude Agent Skills spec:
        - $ARGUMENTS -> full arguments string
        - $1, $2, ... -> individual positional arguments (split on whitespace)
        - ${CLAUDE_SESSION_ID} -> current session ID
        """
        session_id = ctx.config.session_id or ""
        content = content.replace("${CLAUDE_SESSION_ID}", session_id)

        if not arguments:
            content = content.replace("$ARGUMENTS", "")
            content = re.sub(r"\$(\d+)", "", content)
        else:
            content = content.replace("$ARGUMENTS", arguments)
            parts = arguments.split()
            # Replace higher indices first to prevent $1 matching inside $10
            for i in range(len(parts), 0, -1):
                content = content.replace(f"${i}", parts[i - 1])
            # Replace any remaining positional refs with empty
            content = re.sub(r"\$(\d+)", "", content)

        return content

    @staticmethod
    async def _preprocess_dynamic_content(content: str, ctx: ToolContext) -> str:
        """Replace !`command` with command output (dynamic context injection).

        Follows Claude Agent Skills spec: shell commands wrapped in !`...`
        are executed before skill content is sent to the LLM.
        """
        pattern = re.compile(r"!`([^`]+)`")
        matches = list(pattern.finditer(content))
        if not matches:
            return content

        for match in reversed(matches):
            cmd = match.group(1)
            try:
                timeout = min(ctx.engine.config.default_timeout_seconds, 30.0)
                result = await ctx.engine.execute(cmd, timeout=timeout)
                replacement = (
                    result.output.strip() if result.success else f"[Error: {result.error}]"
                )
            except Exception as e:
                logger.warning("Dynamic content injection failed for: %s — %s", cmd, e)
                replacement = f"[Error: {e}]"
            content = content[: match.start()] + replacement + content[match.end() :]

        return content

    async def _execute_skill_forked(
        self, skill: Any, arguments: str, ctx: ToolContext
    ) -> str:
        """Execute skill in a forked (isolated) context.

        Delegates to the AgentRunner's existing fork implementation.
        """
        return await self._runner._execute_skill_forked(skill, arguments)
