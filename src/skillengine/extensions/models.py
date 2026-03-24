"""
Data models for the extension system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Event name constants
SESSION_START = "session_start"
SESSION_END = "session_end"
SKILL_LOADED = "skill_loaded"
SKILL_INVOKED = "skill_invoked"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
COMMAND_INVOKED = "command_invoked"
SNAPSHOT_CREATED = "snapshot_created"


@dataclass
class ExtensionInfo:
    """Metadata about an installed extension."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    source: str = ""  # e.g. "entrypoint", "~/.skillengine/extensions/foo.py"


@dataclass
class ExtensionHook:
    """A registered event hook from an extension."""

    event: str
    handler: Callable[..., Any]
    extension_name: str = ""
    priority: int = 0  # lower runs first


@dataclass
class CommandInfo:
    """A registered slash command."""

    name: str  # e.g. "/quit"
    description: str = ""
    handler: Callable[..., Any] | None = None
    source: str = "builtin"  # "builtin", "extension", "skill", "prompt"
    extension_name: str = ""
    usage: str = ""


@dataclass
class ToolInfo:
    """A tool registered by an extension for LLM function calling."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    extension_name: str = ""
