"""Tests for skill loaders."""

from pathlib import Path
from textwrap import dedent

import pytest

from skillengine.loaders import MarkdownSkillLoader
from skillengine.models import SkillSource


class TestMarkdownSkillLoader:
    """Tests for MarkdownSkillLoader."""

    def test_can_load_skill_md(self, tmp_path: Path) -> None:
        """Should recognize SKILL.md files."""
        loader = MarkdownSkillLoader()

        skill_file = tmp_path / "my-skill" / "SKILL.md"
        skill_file.parent.mkdir()
        skill_file.write_text("# Test")

        assert loader.can_load(skill_file)

    def test_cannot_load_non_md(self, tmp_path: Path) -> None:
        """Should reject non-markdown files."""
        loader = MarkdownSkillLoader()

        txt_file = tmp_path / "skill.txt"
        txt_file.write_text("test")

        assert not loader.can_load(txt_file)

    def test_load_skill_with_frontmatter(self, tmp_path: Path) -> None:
        """Should parse skill with YAML frontmatter."""
        loader = MarkdownSkillLoader()

        skill_dir = tmp_path / "github"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(dedent("""
            ---
            name: github
            description: "GitHub CLI integration"
            metadata:
              emoji: "🐙"
              requires:
                bins:
                  - gh
            ---

            # GitHub Skill

            Use `gh` to interact with GitHub.
        """).strip())

        entry = loader.load_skill(skill_file, SkillSource.WORKSPACE)

        assert entry.skill.name == "github"
        assert entry.skill.description == "GitHub CLI integration"
        assert entry.skill.metadata.emoji == "🐙"
        assert "gh" in entry.skill.metadata.requires.bins
        assert "GitHub Skill" in entry.skill.content

    def test_load_skill_without_frontmatter(self, tmp_path: Path) -> None:
        """Should handle skill without frontmatter."""
        loader = MarkdownSkillLoader()

        skill_dir = tmp_path / "simple"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Simple Skill\n\nJust some instructions.")

        entry = loader.load_skill(skill_file, SkillSource.WORKSPACE)

        assert entry.skill.name == "simple"  # From directory name
        assert "Simple Skill" in entry.skill.content

    def test_load_directory(self, tmp_path: Path) -> None:
        """Should load all skills from a directory."""
        loader = MarkdownSkillLoader()

        # Create two skills
        for name in ["skill-a", "skill-b"]:
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}")

        entries = loader.load_directory(tmp_path, SkillSource.WORKSPACE)

        assert len(entries) == 2
        names = {e.skill.name for e in entries}
        assert names == {"skill-a", "skill-b"}
