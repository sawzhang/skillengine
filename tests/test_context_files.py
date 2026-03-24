"""Tests for context file discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillengine.context_files import (
    CONTEXT_FILE_NAMES,
    ContextFile,
    load_context_files,
)


class TestContextFile:
    """Tests for the ContextFile dataclass."""

    def test_dataclass_creation(self) -> None:
        """Should create a ContextFile with path and content."""
        path = Path("/some/dir/AGENTS.md")
        cf = ContextFile(path=path, content="hello")

        assert cf.path == path
        assert cf.content == "hello"

    def test_dataclass_equality(self) -> None:
        """Two ContextFiles with the same fields should be equal."""
        a = ContextFile(path=Path("/a/AGENTS.md"), content="x")
        b = ContextFile(path=Path("/a/AGENTS.md"), content="x")
        assert a == b


class TestLoadContextFiles:
    """Tests for load_context_files."""

    def test_empty_dirs_return_empty_list(self, tmp_path: Path) -> None:
        """Should return an empty list when no context files exist."""
        result = load_context_files(cwd=tmp_path, home_dir=tmp_path / "fakehome")
        assert result == []

    def test_finds_agents_md_in_home_dir(self, tmp_path: Path) -> None:
        """Should find AGENTS.md in ~/.skillengine/."""
        home_dir = tmp_path / "home"
        sk_dir = home_dir / ".skillengine"
        sk_dir.mkdir(parents=True)
        (sk_dir / "AGENTS.md").write_text("global context")

        cwd = tmp_path / "project"
        cwd.mkdir()

        result = load_context_files(cwd=cwd, home_dir=home_dir)

        assert len(result) == 1
        assert result[0].content == "global context"
        assert result[0].path == (sk_dir / "AGENTS.md").resolve()

    def test_finds_claude_md_in_cwd(self, tmp_path: Path) -> None:
        """Should find CLAUDE.md in the current working directory."""
        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "CLAUDE.md").write_text("project context")

        result = load_context_files(cwd=cwd, home_dir=tmp_path / "fakehome")

        assert len(result) == 1
        assert result[0].content == "project context"
        assert result[0].path == (cwd / "CLAUDE.md").resolve()

    def test_walks_up_directories(self, tmp_path: Path) -> None:
        """Should walk from cwd up to root, collecting context files."""
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)

        (parent / "AGENTS.md").write_text("parent context")
        (child / "AGENTS.md").write_text("child context")

        result = load_context_files(cwd=child, home_dir=tmp_path / "fakehome")

        assert len(result) >= 2
        contents = [cf.content for cf in result]
        assert "parent context" in contents
        assert "child context" in contents
        # Closest files appear last (highest priority)
        assert contents.index("parent context") < contents.index("child context")

    def test_deduplicates_by_resolved_path(self, tmp_path: Path) -> None:
        """Should not return the same file twice even if reachable via different paths."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("only once")

        # Use a symlink to create a second path to the same directory
        link = tmp_path / "link"
        link.symlink_to(project)

        result = load_context_files(cwd=link, home_dir=tmp_path / "fakehome")

        resolved_paths = [cf.path for cf in result]
        # The resolved path should appear only once
        assert resolved_paths.count((project / "AGENTS.md").resolve()) == 1

    def test_agents_md_preferred_over_claude_md_in_same_dir(self, tmp_path: Path) -> None:
        """When both AGENTS.md and CLAUDE.md exist, AGENTS.md should be returned."""
        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AGENTS.md").write_text("agents content")
        (cwd / "CLAUDE.md").write_text("claude content")

        result = load_context_files(cwd=cwd, home_dir=tmp_path / "fakehome")

        # Only one file from cwd since _find_context_file_in_dir returns the first match
        cwd_results = [cf for cf in result if cf.path.parent == cwd.resolve()]
        assert len(cwd_results) == 1
        assert cwd_results[0].content == "agents content"

    def test_closest_files_appear_last(self, tmp_path: Path) -> None:
        """Files closer to cwd should appear later in the list (highest priority)."""
        grandparent = tmp_path / "gp"
        parent = grandparent / "p"
        child = parent / "c"
        child.mkdir(parents=True)

        (grandparent / "AGENTS.md").write_text("grandparent")
        (parent / "AGENTS.md").write_text("parent")
        (child / "AGENTS.md").write_text("child")

        result = load_context_files(cwd=child, home_dir=tmp_path / "fakehome")

        contents = [cf.content for cf in result]
        gp_idx = contents.index("grandparent")
        p_idx = contents.index("parent")
        c_idx = contents.index("child")

        # Furthest first, closest last
        assert gp_idx < p_idx < c_idx

    def test_home_dir_file_appears_before_ancestor_files(self, tmp_path: Path) -> None:
        """Global context from home_dir should appear before ancestor files."""
        home_dir = tmp_path / "home"
        sk_dir = home_dir / ".skillengine"
        sk_dir.mkdir(parents=True)
        (sk_dir / "AGENTS.md").write_text("global")

        cwd = tmp_path / "project"
        cwd.mkdir()
        (cwd / "AGENTS.md").write_text("local")

        result = load_context_files(cwd=cwd, home_dir=home_dir)

        contents = [cf.content for cf in result]
        assert contents.index("global") < contents.index("local")

    def test_context_file_names_constant(self) -> None:
        """CONTEXT_FILE_NAMES should contain the expected filenames."""
        assert "AGENTS.md" in CONTEXT_FILE_NAMES
        assert "CLAUDE.md" in CONTEXT_FILE_NAMES
