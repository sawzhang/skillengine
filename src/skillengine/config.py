"""
Configuration models for the skills engine.

Provides a flexible configuration system that can be loaded from
YAML/JSON files or constructed programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# ---------------------------------------------------------------------------
# Cache retention
# ---------------------------------------------------------------------------

CacheRetention = Literal["none", "short", "long"]


def get_cache_retention() -> CacheRetention:
    """Get cache retention from environment variable, defaulting to 'short'."""
    val = os.environ.get("ASE_CACHE_RETENTION", "short").lower()
    if val in ("none", "short", "long"):
        return val  # type: ignore[return-value]
    return "short"


@dataclass
class SkillEntryConfig:
    """Per-skill configuration overrides."""

    enabled: bool = True  # Enable/disable this skill
    api_key: str | None = None  # API key (applied to primary_env)
    env: dict[str, str] = field(default_factory=dict)  # Env var overrides
    config: dict[str, Any] = field(default_factory=dict)  # Skill-specific config


@dataclass
class SkillsConfig:
    """
    Main configuration for the skills engine.

    Example YAML:
        skill_dirs:
          - ./skills
          - ~/.agent/skills
        bundled_dir: /usr/share/agent/skills
        allow_bundled:
          - github
          - shell
        watch: true
        watch_debounce_ms: 250
        entries:
          github:
            enabled: true
            api_key: "ghp_..."
          private-skill:
            enabled: false
    """

    # Skill loading
    skill_dirs: list[Path] = field(default_factory=list)  # Directories to scan
    bundled_dir: Path | None = None  # Bundled skills directory
    managed_dir: Path | None = None  # User-managed skills (~/.agent/skills)

    # Filtering
    allow_bundled: list[str] | None = None  # Allowlist for bundled skills (None = all)
    exclude_skills: list[str] = field(default_factory=list)  # Skills to exclude

    # File watching
    watch: bool = False  # Enable file watching
    watch_debounce_ms: int = 250  # Debounce delay for file changes

    # Per-skill config
    entries: dict[str, SkillEntryConfig] = field(default_factory=dict)

    # Runtime
    default_timeout_seconds: float = 30.0  # Default execution timeout
    max_concurrent: int = 5  # Max concurrent skill executions

    # Prompt formatting
    prompt_format: str = "xml"  # "xml", "markdown", or "json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillsConfig:
        """Create config from a dictionary."""
        entries = {}
        for name, entry_data in data.get("entries", {}).items():
            entries[name] = SkillEntryConfig(
                enabled=entry_data.get("enabled", True),
                api_key=entry_data.get("api_key"),
                env=entry_data.get("env", {}),
                config=entry_data.get("config", {}),
            )

        return cls(
            skill_dirs=[Path(p) for p in data.get("skill_dirs", [])],
            bundled_dir=Path(data["bundled_dir"]) if data.get("bundled_dir") else None,
            managed_dir=Path(data["managed_dir"]) if data.get("managed_dir") else None,
            allow_bundled=data.get("allow_bundled"),
            exclude_skills=data.get("exclude_skills", []),
            watch=data.get("watch", False),
            watch_debounce_ms=data.get("watch_debounce_ms", 250),
            entries=entries,
            default_timeout_seconds=data.get("default_timeout_seconds", 30.0),
            max_concurrent=data.get("max_concurrent", 5),
            prompt_format=data.get("prompt_format", "xml"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> SkillsConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_yaml_string(cls, content: str) -> SkillsConfig:
        """Load config from a YAML string."""
        data = yaml.safe_load(content)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            "skill_dirs": [str(p) for p in self.skill_dirs],
            "bundled_dir": str(self.bundled_dir) if self.bundled_dir else None,
            "managed_dir": str(self.managed_dir) if self.managed_dir else None,
            "allow_bundled": self.allow_bundled,
            "exclude_skills": self.exclude_skills,
            "watch": self.watch,
            "watch_debounce_ms": self.watch_debounce_ms,
            "entries": {
                name: {
                    "enabled": entry.enabled,
                    "api_key": entry.api_key,
                    "env": entry.env,
                    "config": entry.config,
                }
                for name, entry in self.entries.items()
            },
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_concurrent": self.max_concurrent,
            "prompt_format": self.prompt_format,
        }

    def get_skill_config(self, skill_key: str) -> SkillEntryConfig:
        """Get config for a specific skill, with defaults."""
        return self.entries.get(skill_key, SkillEntryConfig())

    def merge_dirs(self) -> list[Path]:
        """Get all skill directories in priority order (low to high)."""
        dirs: list[Path] = []
        if self.bundled_dir:
            dirs.append(self.bundled_dir)
        if self.managed_dir:
            dirs.append(self.managed_dir)
        dirs.extend(self.skill_dirs)
        return dirs
