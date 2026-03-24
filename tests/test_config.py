"""Tests for configuration models."""

from pathlib import Path
from textwrap import dedent

import pytest

from skillengine.config import SkillEntryConfig, SkillsConfig


class TestSkillsConfig:
    """Tests for SkillsConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = SkillsConfig()

        assert config.skill_dirs == []
        assert config.bundled_dir is None
        assert config.managed_dir is None
        assert config.allow_bundled is None
        assert config.exclude_skills == []
        assert config.watch is False
        assert config.watch_debounce_ms == 250
        assert config.entries == {}
        assert config.default_timeout_seconds == 30.0
        assert config.max_concurrent == 5
        assert config.prompt_format == "xml"

    def test_from_dict_basic(self) -> None:
        """Should create config from a basic dictionary."""
        data = {
            "skill_dirs": ["./skills", "~/.agent/skills"],
            "watch": True,
            "prompt_format": "markdown",
        }

        config = SkillsConfig.from_dict(data)

        assert len(config.skill_dirs) == 2
        assert config.skill_dirs[0] == Path("./skills")
        assert config.skill_dirs[1] == Path("~/.agent/skills")
        assert config.watch is True
        assert config.prompt_format == "markdown"

    def test_from_dict_with_entries(self) -> None:
        """Should create config with skill entries."""
        data = {
            "skill_dirs": ["./skills"],
            "entries": {
                "github": {
                    "enabled": True,
                    "api_key": "ghp_xxx",
                    "env": {"EXTRA_VAR": "value"},
                },
                "disabled-skill": {
                    "enabled": False,
                },
            },
        }

        config = SkillsConfig.from_dict(data)

        assert "github" in config.entries
        assert config.entries["github"].enabled is True
        assert config.entries["github"].api_key == "ghp_xxx"
        assert config.entries["github"].env == {"EXTRA_VAR": "value"}
        assert config.entries["disabled-skill"].enabled is False

    def test_from_dict_with_bundled_managed(self) -> None:
        """Should handle bundled and managed directories."""
        data = {
            "bundled_dir": "/usr/share/agent/skills",
            "managed_dir": "~/.agent/skills",
            "allow_bundled": ["github", "shell"],
            "exclude_skills": ["dangerous-skill"],
        }

        config = SkillsConfig.from_dict(data)

        assert config.bundled_dir == Path("/usr/share/agent/skills")
        assert config.managed_dir == Path("~/.agent/skills")
        assert config.allow_bundled == ["github", "shell"]
        assert config.exclude_skills == ["dangerous-skill"]

    def test_from_dict_empty(self) -> None:
        """Should handle empty dictionary."""
        config = SkillsConfig.from_dict({})

        assert config.skill_dirs == []
        assert config.watch is False

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Should load config from a YAML file."""
        yaml_content = dedent("""
            skill_dirs:
              - ./skills
              - ~/.agent/skills
            watch: true
            watch_debounce_ms: 500
            default_timeout_seconds: 60.0
            entries:
              github:
                enabled: true
                api_key: "ghp_secret"
        """).strip()

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = SkillsConfig.from_yaml(yaml_file)

        assert len(config.skill_dirs) == 2
        assert config.watch is True
        assert config.watch_debounce_ms == 500
        assert config.default_timeout_seconds == 60.0
        assert config.entries["github"].api_key == "ghp_secret"

    def test_from_yaml_string(self) -> None:
        """Should load config from a YAML string."""
        yaml_content = dedent("""
            skill_dirs:
              - ./skills
            prompt_format: json
        """).strip()

        config = SkillsConfig.from_yaml_string(yaml_content)

        assert len(config.skill_dirs) == 1
        assert config.prompt_format == "json"

    def test_from_yaml_string_empty(self) -> None:
        """Should handle empty YAML string."""
        config = SkillsConfig.from_yaml_string("")

        assert config.skill_dirs == []

    def test_to_dict(self) -> None:
        """Should convert config to dictionary."""
        config = SkillsConfig(
            skill_dirs=[Path("./skills")],
            bundled_dir=Path("/usr/share/skills"),
            managed_dir=Path("~/.agent/skills"),
            allow_bundled=["github"],
            exclude_skills=["bad-skill"],
            watch=True,
            watch_debounce_ms=300,
            entries={
                "github": SkillEntryConfig(
                    enabled=True,
                    api_key="ghp_xxx",
                    env={"VAR": "value"},
                    config={"setting": 123},
                )
            },
            default_timeout_seconds=45.0,
            max_concurrent=3,
            prompt_format="markdown",
        )

        data = config.to_dict()

        assert data["skill_dirs"] == ["skills"]  # Path normalizes ./skills to skills
        assert data["bundled_dir"] == "/usr/share/skills"
        assert data["managed_dir"] == "~/.agent/skills"
        assert data["allow_bundled"] == ["github"]
        assert data["exclude_skills"] == ["bad-skill"]
        assert data["watch"] is True
        assert data["watch_debounce_ms"] == 300
        assert data["entries"]["github"]["enabled"] is True
        assert data["entries"]["github"]["api_key"] == "ghp_xxx"
        assert data["entries"]["github"]["env"] == {"VAR": "value"}
        assert data["entries"]["github"]["config"] == {"setting": 123}
        assert data["default_timeout_seconds"] == 45.0
        assert data["max_concurrent"] == 3
        assert data["prompt_format"] == "markdown"

    def test_to_dict_none_values(self) -> None:
        """Should handle None values in to_dict."""
        config = SkillsConfig()

        data = config.to_dict()

        assert data["bundled_dir"] is None
        assert data["managed_dir"] is None
        assert data["allow_bundled"] is None

    def test_roundtrip(self) -> None:
        """Should roundtrip through to_dict/from_dict."""
        original = SkillsConfig(
            skill_dirs=[Path("./skills")],
            watch=True,
            entries={"test": SkillEntryConfig(api_key="key")},
        )

        data = original.to_dict()
        restored = SkillsConfig.from_dict(data)

        assert restored.skill_dirs == original.skill_dirs
        assert restored.watch == original.watch
        assert restored.entries["test"].api_key == original.entries["test"].api_key

    def test_merge_dirs_order(self) -> None:
        """merge_dirs should return directories in priority order."""
        config = SkillsConfig(
            bundled_dir=Path("/usr/share/skills"),
            managed_dir=Path("~/.agent/skills"),
            skill_dirs=[Path("./project/skills"), Path("./extra/skills")],
        )

        dirs = config.merge_dirs()

        # Order: bundled (lowest) -> managed -> skill_dirs (highest)
        assert len(dirs) == 4
        assert dirs[0] == Path("/usr/share/skills")
        assert dirs[1] == Path("~/.agent/skills")
        assert dirs[2] == Path("./project/skills")
        assert dirs[3] == Path("./extra/skills")

    def test_merge_dirs_optional(self) -> None:
        """merge_dirs should skip None directories."""
        config = SkillsConfig(
            bundled_dir=None,
            managed_dir=Path("~/.agent/skills"),
            skill_dirs=[Path("./skills")],
        )

        dirs = config.merge_dirs()

        assert len(dirs) == 2
        assert dirs[0] == Path("~/.agent/skills")
        assert dirs[1] == Path("./skills")

    def test_get_skill_config_exists(self) -> None:
        """Should return skill config when it exists."""
        config = SkillsConfig(
            entries={
                "github": SkillEntryConfig(
                    enabled=True,
                    api_key="ghp_xxx",
                )
            }
        )

        skill_config = config.get_skill_config("github")

        assert skill_config.enabled is True
        assert skill_config.api_key == "ghp_xxx"

    def test_get_skill_config_default(self) -> None:
        """Should return default config when skill not found."""
        config = SkillsConfig()

        skill_config = config.get_skill_config("nonexistent")

        assert skill_config.enabled is True
        assert skill_config.api_key is None
        assert skill_config.env == {}
        assert skill_config.config == {}


class TestSkillEntryConfig:
    """Tests for SkillEntryConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        entry = SkillEntryConfig()

        assert entry.enabled is True
        assert entry.api_key is None
        assert entry.env == {}
        assert entry.config == {}

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        entry = SkillEntryConfig(
            enabled=False,
            api_key="secret",
            env={"VAR": "value"},
            config={"setting": True},
        )

        assert entry.enabled is False
        assert entry.api_key == "secret"
        assert entry.env == {"VAR": "value"}
        assert entry.config == {"setting": True}
