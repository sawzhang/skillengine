"""Tests for skill filters."""

from pathlib import Path

import pytest

from skillengine import SkillsConfig
from skillengine.config import SkillEntryConfig
from skillengine.filters import DefaultSkillFilter
from skillengine.filters.base import FilterContext
from skillengine.models import (
    Skill,
    SkillMetadata,
    SkillRequirements,
    SkillSource,
)


class TestDefaultSkillFilter:
    """Tests for DefaultSkillFilter."""

    def test_basic_skill_eligible(self, sample_skill: Skill) -> None:
        """A skill with no requirements should be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(platform="darwin")

        result = filter.filter(sample_skill, config, context)

        assert result.eligible
        assert result.reason is None

    def test_always_skill_eligible(self, always_skill: Skill) -> None:
        """A skill with always=True should always be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(exclude_skills=["always-skill"])  # Should be ignored
        context = FilterContext(platform="darwin")

        result = filter.filter(always_skill, config, context)

        assert result.eligible

    def test_disabled_skill_ineligible(self, sample_skill: Skill) -> None:
        """A disabled skill should not be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(
            entries={"test-skill": SkillEntryConfig(enabled=False)}
        )
        context = FilterContext(platform="darwin")

        result = filter.filter(sample_skill, config, context)

        assert not result.eligible
        assert "disabled" in result.reason.lower()

    def test_excluded_skill_ineligible(self, sample_skill: Skill) -> None:
        """A skill in the exclusion list should not be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(exclude_skills=["test-skill"])
        context = FilterContext(platform="darwin")

        result = filter.filter(sample_skill, config, context)

        assert not result.eligible
        assert "exclude" in result.reason.lower()

    def test_bundled_allowlist_eligible(self, bundled_skill: Skill) -> None:
        """A bundled skill in the allowlist should be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(allow_bundled=["bundled-skill"])
        context = FilterContext(platform="darwin")

        result = filter.filter(bundled_skill, config, context)

        assert result.eligible

    def test_bundled_not_in_allowlist_ineligible(self, bundled_skill: Skill) -> None:
        """A bundled skill not in the allowlist should not be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(allow_bundled=["other-skill"])
        context = FilterContext(platform="darwin")

        result = filter.filter(bundled_skill, config, context)

        assert not result.eligible
        assert "not in allowlist" in result.reason.lower()

    def test_bundled_no_allowlist_eligible(self, bundled_skill: Skill) -> None:
        """A bundled skill without an allowlist should be eligible."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(allow_bundled=None)  # No allowlist = all allowed
        context = FilterContext(platform="darwin")

        result = filter.filter(bundled_skill, config, context)

        assert result.eligible

    def test_os_requirement_met(self, skill_with_requirements: Skill) -> None:
        """A skill should be eligible when OS requirement is met."""
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            available_bins={"git"},
            env_vars={"GITHUB_TOKEN"},
        )

        result = filter.filter(skill_with_requirements, config, context)

        assert result.eligible

    def test_os_requirement_not_met(self, skill_with_requirements: Skill) -> None:
        """A skill should not be eligible when OS requirement is not met."""
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="win32",
            available_bins={"git"},
            env_vars={"GITHUB_TOKEN"},
        )

        result = filter.filter(skill_with_requirements, config, context)

        assert not result.eligible
        assert "OS" in result.reason

    def test_bins_requirement_met(self) -> None:
        """A skill should be eligible when all required binaries exist."""
        skill = Skill(
            name="bins-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(bins=["git", "docker"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            available_bins={"git", "docker", "python"},
        )

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_bins_requirement_not_met(self) -> None:
        """A skill should not be eligible when a required binary is missing."""
        skill = Skill(
            name="bins-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(bins=["git", "nonexistent-bin"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            available_bins={"git"},
        )

        result = filter.filter(skill, config, context)

        assert not result.eligible
        assert "nonexistent-bin" in result.reason

    def test_any_bins_requirement_met(self) -> None:
        """A skill should be eligible when at least one any_bins binary exists."""
        skill = Skill(
            name="any-bins-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(any_bins=["npm", "pnpm", "yarn"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            available_bins={"pnpm"},  # Only pnpm available
        )

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_any_bins_requirement_not_met(self) -> None:
        """A skill should not be eligible when no any_bins binary exists."""
        skill = Skill(
            name="any-bins-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                # Use nonexistent binaries to ensure they won't be found via PATH
                requires=SkillRequirements(
                    any_bins=["nonexistent-bin-xyz", "fake-tool-abc", "missing-cmd-123"]
                )
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            available_bins={"git"},  # None of the required bins
        )

        result = filter.filter(skill, config, context)

        assert not result.eligible
        assert "nonexistent-bin-xyz" in result.reason or "None of required" in result.reason

    def test_env_requirement_met_in_context(self) -> None:
        """A skill should be eligible when env var is in context."""
        skill = Skill(
            name="env-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(env=["MY_TOKEN"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            env_vars={"MY_TOKEN", "PATH"},
        )

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_env_requirement_met_via_config_override(self) -> None:
        """A skill should be eligible when env var is set via config override."""
        skill = Skill(
            name="env-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(env=["MY_TOKEN"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig(
            entries={"env-skill": SkillEntryConfig(env={"MY_TOKEN": "secret"})}
        )
        context = FilterContext(platform="darwin")

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_env_requirement_met_via_api_key(self) -> None:
        """A skill should be eligible when primary_env matches api_key."""
        skill = Skill(
            name="api-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                primary_env="MY_API_KEY",
                requires=SkillRequirements(env=["MY_API_KEY"]),
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig(
            entries={"api-skill": SkillEntryConfig(api_key="sk-xxx")}
        )
        context = FilterContext(platform="darwin")

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_env_requirement_not_met(self) -> None:
        """A skill should not be eligible when env var is missing."""
        skill = Skill(
            name="env-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(env=["MISSING_VAR"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(platform="darwin")

        result = filter.filter(skill, config, context)

        assert not result.eligible
        assert "MISSING_VAR" in result.reason

    def test_config_requirement_met(self) -> None:
        """A skill should be eligible when config path is truthy."""
        skill = Skill(
            name="config-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(config=["feature.enabled"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            config_values={"feature.enabled": True},
        )

        result = filter.filter(skill, config, context)

        assert result.eligible

    def test_config_requirement_not_met(self) -> None:
        """A skill should not be eligible when config path is falsy."""
        skill = Skill(
            name="config-skill",
            description="Test",
            content="# Test",
            file_path=Path("/tmp/test/SKILL.md"),
            base_dir=Path("/tmp/test"),
            metadata=SkillMetadata(
                requires=SkillRequirements(config=["feature.enabled"])
            ),
        )
        filter = DefaultSkillFilter()
        config = SkillsConfig()
        context = FilterContext(
            platform="darwin",
            config_values={"feature.enabled": False},
        )

        result = filter.filter(skill, config, context)

        assert not result.eligible
        assert "feature.enabled" in result.reason

    def test_filter_all(self, sample_skill: Skill, always_skill: Skill) -> None:
        """filter_all should return results for all skills."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(exclude_skills=["test-skill"])
        context = FilterContext(platform="darwin")

        results = filter.filter_all([sample_skill, always_skill], config, context)

        assert len(results) == 2
        # sample_skill should be excluded
        assert not results[0].eligible
        # always_skill should be eligible
        assert results[1].eligible

    def test_get_eligible(self, sample_skill: Skill, always_skill: Skill) -> None:
        """get_eligible should return only eligible skills."""
        filter = DefaultSkillFilter()
        config = SkillsConfig(exclude_skills=["test-skill"])
        context = FilterContext(platform="darwin")

        eligible = filter.get_eligible([sample_skill, always_skill], config, context)

        assert len(eligible) == 1
        assert eligible[0].name == "always-skill"
