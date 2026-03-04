"""Built-in coding tools for the agent."""

from __future__ import annotations

from skillkit.tools.apply_patch import ApplyPatchTool
from skillkit.tools.bash import BashTool
from skillkit.tools.find import FindTool
from skillkit.tools.grep import GrepTool
from skillkit.tools.ls import LsTool
from skillkit.tools.read import ReadTool
from skillkit.tools.registry import ToolDefinition, ToolRegistry
from skillkit.tools.write import WriteTool

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ReadTool",
    "WriteTool",
    "ApplyPatchTool",
    "BashTool",
    "GrepTool",
    "FindTool",
    "LsTool",
    "create_coding_tools",
    "create_all_tools",
    "create_read_only_tools",
]


def create_coding_tools(cwd: str | None = None) -> list[ToolDefinition]:
    """Create the standard coding tools (read, write, apply_patch, bash)."""
    return [
        ReadTool(cwd).definition(),
        WriteTool(cwd).definition(),
        ApplyPatchTool(cwd).definition(),
        BashTool(cwd).definition(),
    ]


def create_read_only_tools(cwd: str | None = None) -> list[ToolDefinition]:
    """Create read-only tools (read, grep, find, ls)."""
    return [
        ReadTool(cwd).definition(),
        GrepTool(cwd).definition(),
        FindTool(cwd).definition(),
        LsTool(cwd).definition(),
    ]


def create_all_tools(cwd: str | None = None) -> dict[str, ToolDefinition]:
    """Create all built-in tools as a name->definition dict."""
    tools = [
        ReadTool(cwd),
        WriteTool(cwd),
        ApplyPatchTool(cwd),
        BashTool(cwd),
        GrepTool(cwd),
        FindTool(cwd),
        LsTool(cwd),
    ]
    return {t.name: t.definition() for t in tools}
