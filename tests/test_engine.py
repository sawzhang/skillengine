"""Tests for the skills engine."""

from pathlib import Path
from textwrap import dedent

import pytest

from skillengine import SkillsEngine, SkillsConfig


class TestSkillsEngine:
    """Tests for SkillsEngine."""

    def test_load_skills(self, tmp_path: Path) -> None:
        """Should load skills from configured directories."""
        # Create a skill
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(dedent("""
            ---
            name: test-skill
            description: "A test skill"
            ---
            # Test Skill
        """).strip())

        engine = SkillsEngine(
            config=SkillsConfig(skill_dirs=[tmp_path / "skills"])
        )

        skills = engine.load_skills()

        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_filter_skills(self, tmp_path: Path) -> None:
        """Should filter skills based on eligibility."""
        skill_dir = tmp_path / "skills" / "needs-bin"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(dedent("""
            ---
            name: needs-bin
            description: "Needs a non-existent binary"
            metadata:
              requires:
                bins:
                  - nonexistent-binary-xyz
            ---
            # Test
        """).strip())

        engine = SkillsEngine(
            config=SkillsConfig(skill_dirs=[tmp_path / "skills"])
        )

        # Should load but not be eligible
        all_skills = engine.load_skills()
        eligible = engine.filter_skills(all_skills)

        assert len(all_skills) == 1
        assert len(eligible) == 0  # Filtered out due to missing binary

    def test_get_snapshot(self, tmp_path: Path) -> None:
        """Should create and cache skill snapshots."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(dedent("""
            ---
            name: my-skill
            description: "My skill"
            ---
            # My Skill
        """).strip())

        engine = SkillsEngine(
            config=SkillsConfig(skill_dirs=[tmp_path / "skills"])
        )

        snapshot1 = engine.get_snapshot()
        snapshot2 = engine.get_snapshot()

        # Should return cached snapshot
        assert snapshot1 is snapshot2
        assert snapshot1.version == 1

        # Force reload
        snapshot3 = engine.get_snapshot(force_reload=True)
        assert snapshot3 is not snapshot1
        assert snapshot3.version == 2

    def test_format_prompt_xml(self, tmp_path: Path) -> None:
        """Should format skills as XML."""
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(dedent("""
            ---
            name: my-skill
            description: "My awesome skill"
            ---
            # Content
        """).strip())

        engine = SkillsEngine(
            config=SkillsConfig(
                skill_dirs=[tmp_path / "skills"],
                prompt_format="xml",
            )
        )

        snapshot = engine.get_snapshot()

        assert "<skills>" in snapshot.prompt
        assert "<name>my-skill</name>" in snapshot.prompt
        assert "<description>My awesome skill</description>" in snapshot.prompt

    def test_disabled_skill_filtered(self, tmp_path: Path) -> None:
        """Should filter out disabled skills."""
        skill_dir = tmp_path / "skills" / "disabled-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(dedent("""
            ---
            name: disabled-skill
            description: "Should be disabled"
            ---
            # Content
        """).strip())

        from skillengine.config import SkillEntryConfig

        engine = SkillsEngine(
            config=SkillsConfig(
                skill_dirs=[tmp_path / "skills"],
                entries={
                    "disabled-skill": SkillEntryConfig(enabled=False),
                },
            )
        )

        eligible = engine.filter_skills()

        assert len(eligible) == 0


@pytest.mark.asyncio
class TestSkillsEngineAsync:
    """Async tests for SkillsEngine."""

    async def test_execute_command(self) -> None:
        """Should execute shell commands."""
        engine = SkillsEngine()

        result = await engine.execute("echo 'hello world'")

        assert result.success
        assert "hello world" in result.output

    async def test_execute_timeout(self) -> None:
        """Should handle command timeouts."""
        engine = SkillsEngine()

        result = await engine.execute("sleep 10", timeout=0.1)

        assert not result.success
        assert "timed out" in result.error.lower()
