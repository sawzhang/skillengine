"""
Prompt template system for slash commands.

Loads .md files with optional YAML frontmatter as prompt templates
that can be invoked as slash commands with variable substitution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PromptTemplate:
    """A prompt template loaded from a .md file."""

    name: str  # derived from filename (e.g. "review" from "review.md")
    content: str  # raw template content (after frontmatter)
    description: str = ""
    file_path: Path = field(default_factory=lambda: Path())
    variables: list[str] = field(default_factory=list)  # detected variable names


class PromptTemplateLoader:
    """
    Discovers and loads prompt templates from directories.

    Default directories:
    - ~/.skillengine/prompts/
    - ./.skillengine/prompts/
    """

    DEFAULT_DIRS = [
        Path.home() / ".skillengine" / "prompts",
        Path.cwd() / ".skillengine" / "prompts",
    ]

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self.dirs = list(self.DEFAULT_DIRS)
        if extra_dirs:
            self.dirs.extend(extra_dirs)

    def load_all(self) -> list[PromptTemplate]:
        """Load all templates from all configured directories."""
        templates: list[PromptTemplate] = []
        seen_names: set[str] = set()

        for directory in self.dirs:
            if not directory.is_dir():
                continue
            for md_file in sorted(directory.glob("*.md")):
                template = self.load_template(md_file)
                if template and template.name not in seen_names:
                    templates.append(template)
                    seen_names.add(template.name)

        return templates

    def load_template(self, path: Path) -> PromptTemplate | None:
        """Load a single template from a .md file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        name = path.stem  # "review.md" -> "review"
        description = ""
        content = text

        # Parse optional YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1].strip()
                content = parts[2].strip()
                try:
                    frontmatter = yaml.safe_load(frontmatter_str) or {}
                    description = frontmatter.get("description", "")
                except yaml.YAMLError:
                    pass

        # If no description from frontmatter, use first non-empty line
        if not description:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    description = stripped[:80]
                    break

        # Detect variables ($1, $2, $@, ${@:N})
        variables = self._detect_variables(content)

        return PromptTemplate(
            name=name,
            content=content,
            description=description,
            file_path=path,
            variables=variables,
        )

    @staticmethod
    def substitute(template: PromptTemplate, args: str) -> str:
        """
        Substitute variables in a template.

        Variables:
        - $1, $2, ... : positional arguments
        - $@ : all arguments joined
        - ${@:N} : all arguments from position N onward
        """
        parts = args.split() if args else []
        result = template.content

        # Replace ${@:N} first (greedy)
        def replace_slice(m: re.Match[str]) -> str:
            n = int(m.group(1))
            idx = n - 1  # 1-indexed to 0-indexed
            return " ".join(parts[idx:]) if idx < len(parts) else ""

        result = re.sub(r"\$\{@:(\d+)\}", replace_slice, result)

        # Replace $@ with all args
        result = result.replace("$@", " ".join(parts))

        # Replace $1, $2, etc.
        for i, part in enumerate(parts, 1):
            result = result.replace(f"${i}", part)

        return result

    @staticmethod
    def _detect_variables(content: str) -> list[str]:
        """Detect variable references in template content."""
        variables: list[str] = []
        seen: set[str] = set()

        for match in re.finditer(r"\$(\d+|\@|\{@:\d+\})", content):
            var = match.group(0)
            if var not in seen:
                variables.append(var)
                seen.add(var)

        return variables
