"""Tests for CLI commands."""

import json
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from skillengine.cli import cmd_list, cmd_show, cmd_prompt, _create_engine


class MockArgs:
    """Mock argparse namespace for testing."""

    def __init__(self, **kwargs):
        self.dirs = kwargs.get("dirs")
        self.all = kwargs.get("all", False)
        self.json = kwargs.get("json", False)
        self.name = kwargs.get("name")
        self.format = kwargs.get("format", "xml")


class TestCreateEngine:
    """Tests for _create_engine helper."""

    def test_create_with_dirs(self, tmp_path: Path) -> None:
        """Should create engine with specified directories."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()

        engine = _create_engine([str(skill_dir)])

        assert len(engine.config.skill_dirs) == 1
        assert engine.config.skill_dirs[0] == skill_dir

    def test_create_with_default_dir(self) -> None:
        """Should use cwd/skills as default."""
        engine = _create_engine(None)

        assert len(engine.config.skill_dirs) == 1
        assert engine.config.skill_dirs[0].name == "skills"


class TestCmdList:
    """Tests for the list command."""

    def test_list_eligible_skills(self, skill_dir: Path, capsys) -> None:
        """Should list eligible skills."""
        args = MockArgs(dirs=[str(skill_dir)], all=False, json=False)

        cmd_list(args)

        captured = capsys.readouterr()
        # Should show simple-skill (no requirements)
        assert "simple-skill" in captured.out

    def test_list_all_skills(self, skill_dir: Path, capsys) -> None:
        """Should list all skills with --all flag."""
        args = MockArgs(dirs=[str(skill_dir)], all=True, json=False)

        cmd_list(args)

        captured = capsys.readouterr()
        assert "simple-skill" in captured.out
        assert "requires-git" in captured.out
        assert "requires-env" in captured.out

    def test_list_json_output(self, skill_dir: Path, capsys) -> None:
        """Should output JSON with --json flag."""
        args = MockArgs(dirs=[str(skill_dir)], all=True, json=True)

        cmd_list(args)

        captured = capsys.readouterr()
        # Parse the JSON output
        data = json.loads(captured.out)
        names = {s["name"] for s in data}
        assert "simple-skill" in names

    def test_list_empty_dir(self, tmp_path: Path, capsys) -> None:
        """Should handle empty skill directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        args = MockArgs(dirs=[str(empty_dir)], all=False, json=False)

        cmd_list(args)

        captured = capsys.readouterr()
        assert "Total: 0 skills" in captured.out


class TestCmdShow:
    """Tests for the show command."""

    def test_show_skill(self, skill_dir: Path, capsys) -> None:
        """Should show skill details."""
        args = MockArgs(dirs=[str(skill_dir)], name="simple-skill")

        cmd_show(args)

        captured = capsys.readouterr()
        assert "simple-skill" in captured.out
        assert "A simple test skill" in captured.out

    def test_show_skill_with_requirements(self, skill_dir: Path, capsys) -> None:
        """Should show skill requirements."""
        args = MockArgs(dirs=[str(skill_dir)], name="requires-git")

        cmd_show(args)

        captured = capsys.readouterr()
        assert "requires-git" in captured.out
        assert "git" in captured.out

    def test_show_skill_not_found(self, skill_dir: Path) -> None:
        """Should exit with error for nonexistent skill."""
        args = MockArgs(dirs=[str(skill_dir)], name="nonexistent")

        with pytest.raises(SystemExit) as exc_info:
            cmd_show(args)

        assert exc_info.value.code == 1


class TestCmdPrompt:
    """Tests for the prompt command."""

    def test_prompt_xml(self, skill_dir: Path, capsys) -> None:
        """Should generate XML prompt."""
        args = MockArgs(dirs=[str(skill_dir)], format="xml")

        cmd_prompt(args)

        captured = capsys.readouterr()
        assert "<skills>" in captured.out
        assert "<name>" in captured.out

    def test_prompt_markdown(self, skill_dir: Path, capsys) -> None:
        """Should generate Markdown prompt."""
        args = MockArgs(dirs=[str(skill_dir)], format="markdown")

        cmd_prompt(args)

        captured = capsys.readouterr()
        assert "## Available Skills" in captured.out
        assert "**" in captured.out  # Bold formatting

    def test_prompt_json(self, skill_dir: Path, capsys) -> None:
        """Should generate JSON prompt."""
        args = MockArgs(dirs=[str(skill_dir)], format="json")

        cmd_prompt(args)

        captured = capsys.readouterr()
        # Should be valid JSON
        data = json.loads(captured.out)
        assert isinstance(data, list)

    def test_prompt_empty(self, tmp_path: Path, capsys) -> None:
        """Should handle empty prompt."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        args = MockArgs(dirs=[str(empty_dir)], format="xml")

        cmd_prompt(args)

        # Should not crash, may output empty string


class TestXmlEscaping:
    """Tests for XML escaping in prompts."""

    def test_xml_escaping_special_chars(self, tmp_path: Path, capsys) -> None:
        """Should escape special XML characters in skill names and descriptions."""
        skill_dir = tmp_path / "skills" / "special-skill"
        skill_dir.mkdir(parents=True)
        # Use single quotes in YAML to preserve special characters
        (skill_dir / "SKILL.md").write_text(
            dedent("""
            ---
            name: 'test<>skill'
            description: 'A skill with <special> & characters'
            ---
            # Content
        """).strip()
        )

        args = MockArgs(dirs=[str(tmp_path / "skills")], format="xml")

        cmd_prompt(args)

        captured = capsys.readouterr()
        # Check that special characters are escaped
        assert "&lt;" in captured.out  # < escaped
        assert "&gt;" in captured.out  # > escaped
        assert "&amp;" in captured.out  # & escaped
