"""Unified tool dispatcher for all tool types.

Provides a single registration and dispatch point for built-in tools,
skill tools, skill action tools, and extension-registered tools.
Aligns with Managed Agents' `execute(name, input) -> string` interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillengine.tools.apply_patch import ApplyPatchTool
from skillengine.tools.edit import EditTool

logger = logging.getLogger(__name__)

# Type for streaming output callback
OutputCallback = Callable[[str], None]


@dataclass
class ToolContext:
    """Immutable runtime context passed to every tool dispatch call.

    Created per-dispatch by the harness (AgentRunner). Carries the
    current turn's abort signal, engine reference, and config.
    """

    engine: Any  # SkillsEngine — Any to avoid circular import
    config: Any  # AgentConfig — Any to avoid circular import
    abort_event: asyncio.Event
    event_bus: Any  # EventBus
    turn: int = 0
    cwd: str | None = None
    adapter_registry: Any | None = None  # AdapterRegistry
    active_adapter_name: str | None = None
    session_id: str | None = None


class ToolHandler(Protocol):
    """Protocol for tool handlers."""

    async def __call__(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
        on_output: OutputCallback | None = None,
    ) -> str: ...


@dataclass
class _ToolEntry:
    """Internal registration entry."""

    name: str
    handler: Any  # ToolHandler callable
    definition: dict[str, Any]  # OpenAI function-calling format


class ToolDispatcher:
    """Single registry and dispatch point for all tool types.

    All tools — built-in, skill, skill action, extension — are registered
    here and dispatched through a unified ``dispatch()`` method.
    """

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}
        self._pattern_handler: Any | None = None  # handler for ":" pattern (skill actions)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        handler: Any,
        definition: dict[str, Any],
    ) -> None:
        """Register a named tool with its handler and definition."""
        self._tools[name] = _ToolEntry(name=name, handler=handler, definition=definition)

    def register_pattern_handler(self, handler: Any) -> None:
        """Register a handler for pattern-matched tool names (e.g. 'skill:action')."""
        self._pattern_handler = handler

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name. Returns True if found."""
        return self._tools.pop(name, None) is not None

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
        self._pattern_handler = None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
        on_output: OutputCallback | None = None,
    ) -> str:
        """Dispatch a tool call by name.

        Lookup order:
        1. Exact name match in registry
        2. Pattern match via pattern_handler (for ":" separated names)
        3. Error: unknown tool
        """
        # 1. Exact match
        entry = self._tools.get(name)
        if entry is not None:
            return await entry.handler(name, args, ctx, on_output)

        # 2. Pattern match (skill:action)
        if ":" in name and self._pattern_handler is not None:
            return await self._pattern_handler(name, args, ctx, on_output)

        return f"Error: Unknown tool '{name}'"

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI function-calling format."""
        return [entry.definition for entry in self._tools.values()]

    def get_definition(self, name: str) -> dict[str, Any] | None:
        """Get a single tool definition by name."""
        entry = self._tools.get(name)
        return entry.definition if entry else None

    # ------------------------------------------------------------------
    # Bulk registration helpers
    # ------------------------------------------------------------------

    def register_builtins(self, engine: Any) -> None:
        """Register the standard built-in tools (execute, write, read, etc.)."""

        async def _handle_execute(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            command = args.get("command", "")
            cwd = args.get("cwd")
            with ctx.engine.env_context():
                result = await ctx.engine.execute(
                    command,
                    cwd=cwd,
                    on_output=on_output,
                    abort_signal=ctx.abort_event,
                )
            if result.success:
                return result.output or "(no output)"
            return f"Error (exit {result.exit_code}): {result.error}"

        async def _handle_execute_script(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            script = args.get("script", "")
            cwd = args.get("cwd")
            with ctx.engine.env_context():
                result = await ctx.engine.execute_script(
                    script,
                    cwd=cwd,
                    on_output=on_output,
                    abort_signal=ctx.abort_event,
                )
            if result.success:
                return result.output or "(no output)"
            return f"Error (exit {result.exit_code}): {result.error}"

        async def _handle_write(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            file_path = args.get("path", "")
            content = args.get("content", "")
            try:
                p = Path(file_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Written {len(content)} bytes to {file_path}"
            except Exception as e:
                return f"Error writing file: {e}"

        async def _handle_read(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            file_path = args.get("path", "")
            try:
                p = Path(file_path)
                if not p.exists():
                    return f"Error: File not found: {file_path}"
                text = p.read_text(encoding="utf-8")
                if len(text) > 100_000:
                    text = text[:100_000] + "\n... (truncated)"
                return text
            except Exception as e:
                return f"Error reading file: {e}"

        async def _handle_edit(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            workspace_root = ctx.cwd
            tool_cwd = str(workspace_root) if workspace_root else os.getcwd()
            tool = EditTool(cwd=tool_cwd)
            return await tool.execute(args)

        async def _handle_apply_patch(
            name: str,
            args: dict[str, Any],
            ctx: ToolContext,
            on_output: OutputCallback | None = None,
        ) -> str:
            workspace_root = ctx.cwd
            enforce_workspace_boundary = bool(workspace_root)
            tool_cwd = str(workspace_root) if workspace_root else os.getcwd()
            tool = ApplyPatchTool(
                cwd=tool_cwd,
                enforce_workspace_boundary=enforce_workspace_boundary,
            )
            return await tool.execute(args)

        # Register each built-in with its definition
        self.register(
            "execute",
            _handle_execute,
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": (
                        "Execute a shell command and return the output."
                        " Use this to run scripts, CLI tools, or any shell command."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory for the command (optional)",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
        )

        self.register(
            "execute_script",
            _handle_execute_script,
            {
                "type": "function",
                "function": {
                    "name": "execute_script",
                    "description": (
                        "Execute a multi-line shell script."
                        " Use this for complex operations"
                        " that require multiple commands."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "The shell script content to execute",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory for the script (optional)",
                            },
                        },
                        "required": ["script"],
                    },
                },
            },
        )

        self.register(
            "write",
            _handle_write,
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": (
                        "Write content to a file."
                        " Creates parent directories automatically."
                        " Use this instead of heredoc/cat for writing files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The file path to write to",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        )

        self.register(
            "read",
            _handle_read,
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read the contents of a file. Returns the file content as text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The file path to read",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        )

        edit_def = EditTool().definition()
        self.register(
            "edit",
            _handle_edit,
            {
                "type": "function",
                "function": {
                    "name": edit_def.name,
                    "description": edit_def.description,
                    "parameters": edit_def.parameters,
                },
            },
        )

        patch_def = ApplyPatchTool().definition()
        self.register(
            "apply_patch",
            _handle_apply_patch,
            {
                "type": "function",
                "function": {
                    "name": patch_def.name,
                    "description": patch_def.description,
                    "parameters": patch_def.parameters,
                },
            },
        )

    def register_skill_tools(
        self,
        skills: list[Any],
        skill_handler: Any,
    ) -> None:
        """Register the 'skill' meta-tool and per-skill action tools.

        Args:
            skills: List of Skill objects from the engine snapshot.
            skill_handler: Callable handling 'skill' tool dispatch.
        """
        visible_skills = [
            s for s in skills if not s.metadata.invocation.disable_model_invocation
        ]
        if visible_skills:
            skill_names = [s.name for s in visible_skills]
            self.register(
                "skill",
                skill_handler,
                {
                    "type": "function",
                    "function": {
                        "name": "skill",
                        "description": (
                            "Load and execute a skill by name. Skills provide "
                            "specialized capabilities and detailed instructions. "
                            "Call this when the user's request matches a skill's "
                            "description to get the full skill content."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": (
                                        "The skill name to invoke. Available: "
                                        + ", ".join(skill_names)
                                    ),
                                },
                                "arguments": {
                                    "type": "string",
                                    "description": "Optional arguments to pass to the skill",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                },
            )

    def register_action_tools(self, skills: list[Any]) -> None:
        """Register per-skill action tools (skill-name:action-name)."""
        for skill in skills:
            if not skill.has_actions:
                continue
            for action in skill.actions.values():
                tool_name = f"{skill.name}:{action.name}"
                properties: dict[str, Any] = {}
                required: list[str] = []
                for param in action.params:
                    prop: dict[str, Any] = {"description": param.description or param.name}
                    if param.type == "file":
                        prop["type"] = "string"
                        prop["description"] = (param.description or param.name) + " (file path)"
                    elif param.type == "number":
                        prop["type"] = "number"
                    elif param.type == "bool":
                        prop["type"] = "boolean"
                    else:
                        prop["type"] = "string"
                    if param.default is not None:
                        prop["default"] = param.default
                    properties[param.name] = prop
                    if param.required:
                        required.append(param.name)

                # Action tools use the pattern handler, so we register definition only
                # for get_definitions() output. Dispatch goes through pattern_handler.
                self.register(
                    tool_name,
                    self._pattern_handler or (lambda *a, **k: "Error: no action handler"),
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": action.description or f"{skill.name} {action.name}",
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            },
                        },
                    },
                )

    def register_extension_tools(self, extensions: Any) -> None:
        """Register tools from the extension system.

        Args:
            extensions: ExtensionManager with get_tools() method.
        """
        if not extensions:
            return
        for tool_info in extensions.get_tools():
            handler = tool_info.handler

            async def _handle_extension(
                name: str,
                args: dict[str, Any],
                ctx: ToolContext,
                on_output: OutputCallback | None = None,
                _handler: Any = handler,
            ) -> str:
                result = _handler(**args)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result) if result is not None else "(no output)"

            self.register(
                tool_info.name,
                _handle_extension,
                {
                    "type": "function",
                    "function": {
                        "name": tool_info.name,
                        "description": tool_info.description,
                        "parameters": tool_info.parameters,
                    },
                },
            )
