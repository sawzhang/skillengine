"""
Base skill loader interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from skillengine.models import SkillEntry, SkillSource


class SkillLoader(ABC):
    """
    Abstract base class for skill loaders.

    Implement this interface to support loading skills from different
    file formats (Markdown, YAML, JSON, etc.).
    """

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass

    @abstractmethod
    def load_skill(self, path: Path, source: SkillSource) -> SkillEntry:
        """Load a skill from a file."""
        pass

    def load_directory(
        self,
        directory: Path,
        source: SkillSource,
        recursive: bool = True,
    ) -> list[SkillEntry]:
        """
        Load all skills from a directory.

        Args:
            directory: Directory to scan
            source: Origin of these skills
            recursive: Whether to scan subdirectories

        Returns:
            List of loaded skill entries
        """
        if not directory.exists():
            return []

        entries: list[SkillEntry] = []

        if recursive:
            # Look for SKILL.md in subdirectories
            for skill_dir in directory.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and self.can_load(skill_file):
                        entries.append(self.load_skill(skill_file, source))
        else:
            # Look for .md files directly
            for path in directory.glob("*.md"):
                if self.can_load(path):
                    entries.append(self.load_skill(path, source))

        return entries

    def load_directories(
        self,
        directories: list[tuple[Path, SkillSource]],
        recursive: bool = True,
    ) -> dict[str, SkillEntry]:
        """
        Load skills from multiple directories with precedence.

        Later directories override earlier ones for skills with the same name.

        Args:
            directories: List of (path, source) tuples in priority order
            recursive: Whether to scan subdirectories

        Returns:
            Dictionary of skill name -> entry (merged by precedence)
        """
        merged: dict[str, SkillEntry] = {}

        for directory, source in directories:
            entries = self.load_directory(directory, source, recursive)
            for entry in entries:
                if entry.skill:
                    merged[entry.skill.name] = entry

        return merged
