"""Package data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathMetadata:
    """Metadata tracking where a resource came from."""

    source: str = "local"  # "local", "pypi", "git", "cli"
    scope: str = "project"  # "user", "project", "temporary"
    origin: str = "top-level"  # "package", "top-level"
    base_dir: str = ""


@dataclass
class ResolvedResource:
    """A resolved resource path with metadata."""

    path: Path
    enabled: bool = True
    metadata: PathMetadata = field(default_factory=PathMetadata)


@dataclass
class PackageManifest:
    """Defines the contents of a skill package.

    Can be specified in pyproject.toml under [tool.skillengine]::

        [tool.skillengine]
        extensions = ["./ext.py"]
        skills = ["./skills/**/*.md"]
        themes = ["./themes/**/*.json"]
        prompts = ["./prompts/**/*.md"]
    """

    extensions: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageManifest:
        """Create from a dictionary (e.g., from pyproject.toml)."""
        return cls(
            extensions=data.get("extensions", []),
            skills=data.get("skills", []),
            themes=data.get("themes", []),
            prompts=data.get("prompts", []),
        )

    @property
    def is_empty(self) -> bool:
        """Check if the manifest has no resources."""
        return not (self.extensions or self.skills or self.themes or self.prompts)


@dataclass
class ResolvedPackage:
    """A package with resolved resource paths."""

    name: str
    version: str = ""
    source_type: str = "local"  # "local", "pypi", "git"
    base_dir: Path = field(default_factory=lambda: Path("."))
    manifest: PackageManifest = field(default_factory=PackageManifest)
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
