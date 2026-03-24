"""Tests for the prompt template system."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from skillengine.prompts import PromptTemplate, PromptTemplateLoader


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temp directory with prompt templates."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()

    # Template with YAML frontmatter
    (prompts / "review.md").write_text(
        dedent("""
        ---
        description: "Review code changes focusing on a specific area"
        ---
        Review the following code changes, focusing on $1.

        $@
        """).strip()
    )

    # Template without frontmatter
    (prompts / "explain.md").write_text(
        dedent("""
        Explain the following code in detail.

        $@
        """).strip()
    )

    # Template with slice variable
    (prompts / "summarize.md").write_text(
        dedent("""
        ---
        description: "Summarize content"
        ---
        Summarize $1 from the following:

        ${@:2}
        """).strip()
    )

    return prompts


@pytest.fixture
def loader(prompts_dir: Path) -> PromptTemplateLoader:
    return PromptTemplateLoader(extra_dirs=[prompts_dir])


class TestPromptTemplateLoading:
    def test_load_all(self, loader: PromptTemplateLoader) -> None:
        templates = loader.load_all()
        names = {t.name for t in templates}
        assert "review" in names
        assert "explain" in names
        assert "summarize" in names

    def test_load_template_with_frontmatter(self, prompts_dir: Path) -> None:
        loader = PromptTemplateLoader()
        template = loader.load_template(prompts_dir / "review.md")
        assert template is not None
        assert template.name == "review"
        assert template.description == "Review code changes focusing on a specific area"
        assert "$1" in template.content
        assert "---" not in template.content

    def test_load_template_without_frontmatter(self, prompts_dir: Path) -> None:
        loader = PromptTemplateLoader()
        template = loader.load_template(prompts_dir / "explain.md")
        assert template is not None
        assert template.name == "explain"
        # Description from first non-heading line
        assert "Explain" in template.description

    def test_load_template_nonexistent(self) -> None:
        loader = PromptTemplateLoader()
        template = loader.load_template(Path("/nonexistent/file.md"))
        assert template is None

    def test_detected_variables(self, prompts_dir: Path) -> None:
        loader = PromptTemplateLoader()
        template = loader.load_template(prompts_dir / "review.md")
        assert template is not None
        assert "$1" in template.variables
        assert "$@" in template.variables

    def test_detected_slice_variables(self, prompts_dir: Path) -> None:
        loader = PromptTemplateLoader()
        template = loader.load_template(prompts_dir / "summarize.md")
        assert template is not None
        assert "$1" in template.variables
        assert "${@:2}" in template.variables

    def test_load_from_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loader = PromptTemplateLoader(extra_dirs=[empty_dir])
        # Default dirs likely don't exist in test, so we get empty
        templates = loader.load_all()
        assert isinstance(templates, list)

    def test_no_duplicate_names(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir1 / "test.md").write_text("First version")
        (dir2 / "test.md").write_text("Second version")
        loader = PromptTemplateLoader(extra_dirs=[dir1, dir2])
        templates = loader.load_all()
        test_templates = [t for t in templates if t.name == "test"]
        assert len(test_templates) == 1


class TestVariableSubstitution:
    def test_positional_args(self) -> None:
        template = PromptTemplate(name="test", content="Hello $1 and $2!")
        result = PromptTemplateLoader.substitute(template, "Alice Bob")
        assert result == "Hello Alice and Bob!"

    def test_all_args(self) -> None:
        template = PromptTemplate(name="test", content="Args: $@")
        result = PromptTemplateLoader.substitute(template, "one two three")
        assert result == "Args: one two three"

    def test_slice_args(self) -> None:
        template = PromptTemplate(name="test", content="First: $1, Rest: ${@:2}")
        result = PromptTemplateLoader.substitute(template, "one two three")
        assert result == "First: one, Rest: two three"

    def test_empty_args(self) -> None:
        template = PromptTemplate(name="test", content="Hello $1!")
        result = PromptTemplateLoader.substitute(template, "")
        assert result == "Hello $1!"

    def test_excess_placeholders(self) -> None:
        template = PromptTemplate(name="test", content="$1 $2 $3")
        result = PromptTemplateLoader.substitute(template, "only")
        assert "only" in result
        assert "$2" in result  # not substituted

    def test_combined(self) -> None:
        template = PromptTemplate(
            name="test",
            content="Focus on $1.\n\nContext: $@\n\nExtra: ${@:2}",
        )
        result = PromptTemplateLoader.substitute(template, "security auth tokens")
        assert "Focus on security." in result
        assert "Context: security auth tokens" in result
        assert "Extra: auth tokens" in result


class TestDetectVariables:
    def test_detect_simple(self) -> None:
        vars = PromptTemplateLoader._detect_variables("Hello $1 and $2")
        assert vars == ["$1", "$2"]

    def test_detect_all_args(self) -> None:
        vars = PromptTemplateLoader._detect_variables("$@")
        assert vars == ["$@"]

    def test_detect_slice(self) -> None:
        vars = PromptTemplateLoader._detect_variables("${@:2}")
        assert vars == ["${@:2}"]

    def test_no_duplicates(self) -> None:
        vars = PromptTemplateLoader._detect_variables("$1 $1 $1")
        assert vars == ["$1"]

    def test_no_variables(self) -> None:
        vars = PromptTemplateLoader._detect_variables("no variables here")
        assert vars == []
