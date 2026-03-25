"""Claude Agent SDK Bridge — run SkillEngine Skills via Claude Agent SDK.

Enables dual-mode execution:
- Mode 1: SkillEngine's native AgentRunner (own agent loop)
- Mode 2: Claude Agent SDK query() (Claude's agent loop)

Same SKILL.md asset, two runtime paths.

Requires: pip install claude-agent-sdk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skillengine.engine import SkillsEngine
    from skillengine.models import Skill

logger = logging.getLogger(__name__)


class ClaudeSDKBridge:
    """Bridge SkillEngine Skills to Claude Agent SDK execution.

    Usage::

        from skillengine import SkillsEngine, SkillsConfig
        from skillengine.a2a.claude_sdk_bridge import ClaudeSDKBridge

        engine = SkillsEngine(config=SkillsConfig(skill_dirs=["./skills"]))
        bridge = ClaudeSDKBridge(engine)

        # Run a skill via Claude Agent SDK
        result = await bridge.run_skill("read-tweet", "https://x.com/...")

        # Export skill as SDK agent definition
        agent_def = bridge.to_sdk_agent_definition("mcp-review")
    """

    def __init__(self, engine: SkillsEngine) -> None:
        self.engine = engine

    async def run_skill(
        self,
        skill_name: str,
        input_text: str,
        *,
        max_turns: int = 20,
        cwd: str | Path | None = None,
        permission_mode: str = "acceptEdits",
        model: str | None = None,
        **extra_options: Any,
    ) -> str:
        """Execute a Skill via Claude Agent SDK's query() function.

        The SKILL.md content becomes the system prompt,
        allowed-tools maps to SDK's allowed_tools,
        and model maps to SDK's model override.

        Args:
            skill_name: Name of the skill to execute.
            input_text: User input / task description.
            max_turns: Maximum conversation turns.
            cwd: Working directory for tool execution.
            permission_mode: SDK permission mode.
            model: Model override (defaults to skill's model field).
            **extra_options: Additional ClaudeAgentOptions fields.

        Returns:
            Combined text output from the SDK execution.

        Raises:
            ValueError: If skill not found.
            ImportError: If claude-agent-sdk is not installed.
        """
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "Claude Agent SDK is required. Install with: pip install claude-agent-sdk"
            )

        skill = self._get_skill(skill_name)

        # Build SDK options from Skill definition
        options_kwargs: dict[str, Any] = {
            "system_prompt": skill.content,
            "permission_mode": permission_mode,
            "max_turns": max_turns,
        }

        # Map allowed-tools
        if skill.allowed_tools:
            options_kwargs["allowed_tools"] = skill.allowed_tools

        # Map model (skill-level override → parameter override)
        effective_model = model or skill.model
        if effective_model:
            options_kwargs["model"] = effective_model

        # Working directory
        if cwd:
            options_kwargs["cwd"] = Path(cwd)

        # Merge extra options
        options_kwargs.update(extra_options)

        options = ClaudeAgentOptions(**options_kwargs)

        # Execute via SDK
        result_parts: list[str] = []
        async for message in query(prompt=input_text, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_parts.append(block.text)

        output = "\n".join(result_parts)
        logger.debug(
            "SDK execution of skill '%s': %d chars output",
            skill_name,
            len(output),
        )
        return output

    def to_sdk_agent_definition(
        self,
        skill_name: str,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Export a Skill as a Claude Agent SDK AgentDefinition.

        The returned dict can be injected into
        ClaudeAgentOptions.agents for sub-agent orchestration.

        Args:
            skill_name: Name of the skill to export.
            model_override: Override the skill's model setting.

        Returns:
            Dict matching Claude Agent SDK's agent definition format.
        """
        skill = self._get_skill(skill_name)

        definition: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.content,
        }

        if skill.allowed_tools:
            definition["tools"] = skill.allowed_tools

        model = model_override or skill.model
        if model:
            definition["model"] = model

        return definition

    def to_sdk_agents(
        self,
        skill_names: list[str] | None = None,
        model_override: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Export multiple Skills as SDK agent definitions.

        Args:
            skill_names: Skills to export (None = all skills).
            model_override: Override model for all agents.

        Returns:
            Dict of {name: agent_definition} for ClaudeAgentOptions.agents.
        """
        if skill_names is None:
            skills = self.engine.load_skills()
            skill_names = [s.name for s in skills]

        agents = {}
        for name in skill_names:
            try:
                agents[name] = self.to_sdk_agent_definition(name, model_override)
            except ValueError:
                logger.warning("Skill '%s' not found, skipping", name)

        return agents

    def to_mcp_tool(self, skill_name: str) -> dict[str, Any]:
        """Export a Skill as an MCP tool definition.

        Can be used with Claude Agent SDK's create_sdk_mcp_server()
        to expose a skill as an in-process MCP tool.

        Args:
            skill_name: Name of the skill.

        Returns:
            MCP tool definition dict.
        """
        skill = self._get_skill(skill_name)

        # Extract input schema from a2a config if available
        a2a = getattr(skill, "_a2a_config", {})
        input_schema = a2a.get(
            "input_schema",
            {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Task input or query",
                    },
                },
                "required": ["input"],
            },
        )

        return {
            "name": skill.name,
            "description": skill.description,
            "inputSchema": input_schema,
        }

    def _get_skill(self, skill_name: str) -> Skill:
        """Get a skill by name, raising if not found."""
        skill = self.engine.get_skill(skill_name)
        if skill is None:
            available = [s.name for s in self.engine.load_skills()]
            raise ValueError(f"Skill '{skill_name}' not found. Available: {available}")
        return skill
