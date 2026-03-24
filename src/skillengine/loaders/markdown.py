"""
Markdown skill loader with YAML frontmatter support.

Skill files should follow this format:

```markdown
---
name: my-skill
description: "A brief description"
metadata:
  emoji: "🔧"
  requires:
    bins: ["some-cli"]
    env: ["API_KEY"]
  primary_env: "API_KEY"
user-invocable: true
---

# My Skill

Detailed instructions for the LLM on how to use this skill...
```
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from skillengine.loaders.base import SkillLoader
from skillengine.models import (
    InstallKind,
    Skill,
    SkillAction,
    SkillActionParam,
    SkillEntry,
    SkillInstallSpec,
    SkillInvocationPolicy,
    SkillMetadata,
    SkillRequirements,
    SkillSource,
)

# Regex to match YAML frontmatter
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


class MarkdownSkillLoader(SkillLoader):
    """
    Loads skills from Markdown files with YAML frontmatter.

    Expected file structure:
    - skills/<skill-name>/SKILL.md
    - Or: skills/<skill-name>.md
    """

    def can_load(self, path: Path) -> bool:
        """Check if this is a valid skill file."""
        if not path.exists():
            return False
        if path.suffix.lower() != ".md":
            return False
        # Must be named SKILL.md or be a .md file in skills dir
        return path.name == "SKILL.md" or path.parent.name == "skills"

    def load_skill(self, path: Path, source: SkillSource) -> SkillEntry:
        """Load a skill from a Markdown file."""
        try:
            content = path.read_text(encoding="utf-8")
            return self._parse_skill_file(path, content, source)
        except Exception as e:
            # Return entry with error
            return SkillEntry(
                skill=Skill(
                    name=path.stem,
                    description="",
                    content="",
                    file_path=path,
                    base_dir=path.parent,
                    source=source,
                ),
                raw_content="",
                load_error=str(e),
            )

    def _parse_skill_file(
        self,
        path: Path,
        content: str,
        source: SkillSource,
    ) -> SkillEntry:
        """Parse a skill file and extract frontmatter."""
        frontmatter: dict[str, Any] = {}
        body = content

        # Extract frontmatter if present
        match = FRONTMATTER_PATTERN.match(content)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = content[match.end() :]

        # Determine skill name
        name = frontmatter.get("name")
        if not name:
            # Use directory name or file stem
            if path.name == "SKILL.md":
                name = path.parent.name
            else:
                name = path.stem

        # Extract description
        description = frontmatter.get("description", "")
        if not description:
            # Try to extract from first paragraph
            lines = body.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    description = line[:200]
                    break

        # Parse metadata
        metadata = self._parse_metadata(frontmatter)

        # Parse actions
        actions = self._parse_actions(frontmatter)

        # Parse Claude Agent Skills extensions from frontmatter
        allowed_tools = self._ensure_list(frontmatter.get("allowed-tools", []))
        skill_model = frontmatter.get("model")
        context = frontmatter.get("context")
        argument_hint = frontmatter.get("argument-hint")
        raw_hooks = frontmatter.get("hooks")
        hooks = raw_hooks if isinstance(raw_hooks, dict) else {}

        skill = Skill(
            name=name,
            description=description,
            content=body.strip(),
            file_path=path,
            base_dir=path.parent,
            source=source,
            metadata=metadata,
            actions=actions,
            allowed_tools=allowed_tools,
            model=skill_model if isinstance(skill_model, str) else None,
            context=context if isinstance(context, str) else None,
            argument_hint=argument_hint if isinstance(argument_hint, str) else None,
            hooks={str(k): str(v) for k, v in hooks.items()},
        )

        # Attach A2A config if present (used by AgentCard.from_skill)
        a2a_config = frontmatter.get("a2a", {})
        if isinstance(a2a_config, dict) and a2a_config:
            skill._a2a_config = a2a_config  # type: ignore[attr-defined]

        return SkillEntry(
            skill=skill,
            frontmatter=frontmatter,
            raw_content=content,
        )

    def _parse_metadata(self, frontmatter: dict[str, Any]) -> SkillMetadata:
        """Parse metadata from frontmatter."""
        raw_metadata = frontmatter.get("metadata", {})

        # Handle different metadata formats
        if isinstance(raw_metadata, str):
            try:
                raw_metadata = yaml.safe_load(raw_metadata) or {}
            except yaml.YAMLError:
                raw_metadata = {}

        # Check for nested metadata (e.g., {"openclaw": {...}})
        if "openclaw" in raw_metadata:
            raw_metadata = raw_metadata["openclaw"]
        elif "manifest" in raw_metadata:
            raw_metadata = raw_metadata["manifest"]

        # Parse requirements
        requires_raw = raw_metadata.get("requires", {})
        any_bins_raw = requires_raw.get("anyBins", requires_raw.get("any_bins", []))
        requires = SkillRequirements(
            bins=self._ensure_list(requires_raw.get("bins", [])),
            any_bins=self._ensure_list(any_bins_raw),
            env=self._ensure_list(requires_raw.get("env", [])),
            config=self._ensure_list(requires_raw.get("config", [])),
            os=self._ensure_list(raw_metadata.get("os", [])),
        )

        # Parse install specs
        install_specs: list[SkillInstallSpec] = []
        for spec in raw_metadata.get("install", []):
            if isinstance(spec, dict):
                install_specs.append(self._parse_install_spec(spec))

        # Parse invocation policy
        invocation = SkillInvocationPolicy(
            user_invocable=frontmatter.get("user-invocable", True),
            disable_model_invocation=frontmatter.get("disable-model-invocation", False),
            require_confirmation=frontmatter.get("require-confirmation", False),
        )

        return SkillMetadata(
            always=raw_metadata.get("always", False),
            skill_key=raw_metadata.get("skillKey", raw_metadata.get("skill_key")),
            primary_env=raw_metadata.get("primaryEnv", raw_metadata.get("primary_env")),
            memory_scope=raw_metadata.get("memoryScope", raw_metadata.get("memory_scope")),
            emoji=raw_metadata.get("emoji"),
            homepage=raw_metadata.get("homepage"),
            author=raw_metadata.get("author"),
            version=raw_metadata.get("version"),
            tags=self._ensure_list(raw_metadata.get("tags", [])),
            requires=requires,
            install=install_specs,
            invocation=invocation,
        )

    def _parse_install_spec(self, spec: dict[str, Any]) -> SkillInstallSpec:
        """Parse an installation specification."""
        kind_str = spec.get("kind", spec.get("id", "download"))
        try:
            kind = InstallKind(kind_str.lower())
        except ValueError:
            kind = InstallKind.DOWNLOAD

        return SkillInstallSpec(
            kind=kind,
            id=spec.get("id"),
            label=spec.get("label"),
            bins=self._ensure_list(spec.get("bins", [])),
            os=self._ensure_list(spec.get("os", [])),
            url=spec.get("url"),
            args=self._ensure_list(spec.get("args", [])),
        )

    def _parse_actions(self, frontmatter: dict[str, Any]) -> dict[str, SkillAction]:
        """Parse actions from frontmatter."""
        raw_actions = frontmatter.get("actions", {})
        if not isinstance(raw_actions, dict):
            return {}

        actions: dict[str, SkillAction] = {}
        for action_name, action_data in raw_actions.items():
            if not isinstance(action_data, dict):
                continue
            script = action_data.get("script", "")
            if not script:
                continue

            params: list[SkillActionParam] = []
            for param_name, param_data in action_data.get("params", {}).items():
                if isinstance(param_data, dict):
                    params.append(
                        SkillActionParam(
                            name=param_name,
                            type=param_data.get("type", "string"),
                            required=param_data.get("required", False),
                            position=param_data.get("position"),
                            description=param_data.get("description", ""),
                            default=param_data.get("default"),
                        )
                    )
                else:
                    # Simple form: param_name: type
                    params.append(SkillActionParam(name=param_name, type=str(param_data)))

            actions[action_name] = SkillAction(
                name=action_name,
                script=script,
                description=action_data.get("description", ""),
                params=params,
                output=action_data.get("output", "text"),
            )

        return actions

    @staticmethod
    def _ensure_list(value: Any) -> list[str]:
        """Ensure value is a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value]
        return []
