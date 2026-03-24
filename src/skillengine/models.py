"""
Core data models for the skills engine.

These models define the structure of skills, their metadata, requirements,
and runtime state. They are designed to be serializable and framework-agnostic.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Content types for multi-modal messages
# ---------------------------------------------------------------------------


@dataclass
class TextContent:
    """Text content block within a message."""

    type: str = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content block within a message (base64-encoded)."""

    type: str = "image"
    data: str = ""  # base64-encoded image data
    mime_type: str = "image/png"  # e.g. image/png, image/jpeg, image/gif, image/webp


# Union type for message content
MessageContent = str | list[TextContent | ImageContent]


class SkillSource(str, Enum):
    """Origin of a skill definition."""

    BUNDLED = "bundled"  # Shipped with the application
    MANAGED = "managed"  # User-installed (e.g., ~/.agent/skills)
    WORKSPACE = "workspace"  # Project-local (e.g., ./skills)
    PLUGIN = "plugin"  # From a plugin/extension
    EXTRA = "extra"  # From extra directories


class InstallKind(str, Enum):
    """Type of installation method."""

    BREW = "brew"
    APT = "apt"
    NPM = "npm"
    PIP = "pip"
    UV = "uv"
    GO = "go"
    CARGO = "cargo"
    DOWNLOAD = "download"


@dataclass
class SkillInstallSpec:
    """Installation specification for a skill's dependencies."""

    kind: InstallKind
    id: str | None = None  # Package/formula name
    label: str | None = None  # Display label
    bins: list[str] = field(default_factory=list)  # Produced binaries
    os: list[str] = field(default_factory=list)  # Applicable platforms
    url: str | None = None  # For download kind
    args: list[str] = field(default_factory=list)  # Extra install args


@dataclass
class SkillRequirements:
    """Requirements that must be satisfied for a skill to be eligible."""

    bins: list[str] = field(default_factory=list)  # All must exist
    any_bins: list[str] = field(default_factory=list)  # At least one must exist
    env: list[str] = field(default_factory=list)  # Required env vars
    config: list[str] = field(default_factory=list)  # Required config paths
    os: list[str] = field(default_factory=list)  # Supported platforms


@dataclass
class SkillActionParam:
    """A parameter for a skill action."""

    name: str
    type: str = "string"  # string, file, json, number, bool
    required: bool = False
    position: int | None = None  # positional arg index (1-based)
    description: str = ""
    default: str | None = None


@dataclass
class SkillAction:
    """
    A deterministic action that can be executed without LLM.

    Maps to a script file in the skill's directory that accepts CLI arguments.
    """

    name: str  # e.g. "extract-fields"
    script: str  # relative path from skill base_dir, e.g. "scripts/extract_form_field_info.py"
    description: str = ""
    params: list[SkillActionParam] = field(default_factory=list)
    output: str = "text"  # text, json, file


@dataclass
class SkillInvocationPolicy:
    """Controls how a skill can be invoked."""

    user_invocable: bool = True  # Can user invoke via /skill-name
    disable_model_invocation: bool = False  # Hide from LLM system prompt
    require_confirmation: bool = False  # Require user confirmation before execution


@dataclass
class SkillMetadata:
    """Extended metadata for a skill."""

    always: bool = False  # Always include (override eligibility)
    skill_key: str | None = None  # Custom unique key for config lookups
    primary_env: str | None = None  # Primary environment variable (for API keys)
    emoji: str | None = None  # Visual indicator
    homepage: str | None = None  # Project homepage URL
    author: str | None = None  # Skill author
    version: str | None = None  # Skill version
    tags: list[str] = field(default_factory=list)  # Categorization tags
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    install: list[SkillInstallSpec] = field(default_factory=list)
    invocation: SkillInvocationPolicy = field(default_factory=SkillInvocationPolicy)


@dataclass
class Skill:
    """
    A skill definition loaded from a SKILL.md file.

    Skills are capabilities that can be made available to an LLM agent.
    They consist of a name, description, and optional metadata that
    controls eligibility and behavior.
    """

    name: str  # Unique identifier
    description: str  # One-line description for LLM
    content: str  # Full skill content (instructions for LLM)
    file_path: Path  # Full path to SKILL.md
    base_dir: Path  # Parent directory (for relative path resolution)
    source: SkillSource = SkillSource.WORKSPACE
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    actions: dict[str, SkillAction] = field(default_factory=dict)

    # Claude Agent Skills extensions
    allowed_tools: list[str] = field(default_factory=list)  # Per-skill tool restrictions
    model: str | None = None  # Per-skill model override
    context: str | None = None  # "fork" for isolated subagent execution
    argument_hint: str | None = None  # Autocomplete hint for slash commands
    hooks: dict[str, str] = field(default_factory=dict)  # Per-skill lifecycle hooks

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Skill):
            return False
        return self.name == other.name

    @property
    def skill_key(self) -> str:
        """Get the config lookup key for this skill."""
        return self.metadata.skill_key or self.name

    def content_hash(self) -> str:
        """Generate a hash of the skill content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def get_action(self, name: str) -> SkillAction | None:
        """Get an action by name."""
        return self.actions.get(name)

    @property
    def has_actions(self) -> bool:
        """Check if this skill has deterministic actions."""
        return bool(self.actions)


@dataclass
class SkillEntry:
    """
    A skill entry with parsed frontmatter and resolved metadata.

    This is the internal representation used during loading and filtering.
    """

    skill: Skill
    frontmatter: dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""  # Original file content
    load_error: str | None = None  # Error message if loading failed


@dataclass
class SkillSnapshot:
    """
    A point-in-time snapshot of eligible skills.

    Used for caching and to detect when skills need to be reloaded.
    """

    skills: list[Skill]
    prompt: str  # Pre-formatted prompt for LLM
    version: int  # Snapshot version for cache invalidation
    timestamp: float = field(default_factory=time.time)
    source_dirs: list[Path] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.version, self.timestamp))

    @property
    def skill_names(self) -> list[str]:
        """Get list of skill names in this snapshot."""
        return [s.name for s in self.skills]

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None
