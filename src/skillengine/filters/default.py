"""
Default skill filter implementation.

Implements the standard eligibility checks:
1. Skill enabled in config
2. Bundled skill allowlist
3. OS requirements
4. Required binaries
5. Required environment variables
6. Required config paths
7. Always-include override
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import TYPE_CHECKING

from skillengine.config import SkillsConfig
from skillengine.filters.base import FilterContext, FilterResult, SkillFilter
from skillengine.models import Skill, SkillSource

if TYPE_CHECKING:
    from skillengine.config import SkillEntryConfig


class DefaultSkillFilter(SkillFilter):
    """
    Default implementation of skill filtering.

    Checks multiple criteria in order and returns the first failure reason.
    """

    def filter(
        self,
        skill: Skill,
        config: SkillsConfig,
        context: FilterContext,
    ) -> FilterResult:
        """Check if a skill is eligible."""
        # 1. Check if always included
        if skill.metadata.always:
            return FilterResult(skill=skill, eligible=True)

        # 2. Check if explicitly disabled
        skill_config = config.get_skill_config(skill.skill_key)
        if not skill_config.enabled:
            return FilterResult(
                skill=skill,
                eligible=False,
                reason=f"Skill '{skill.name}' is disabled in config",
            )

        # 3. Check exclusion list
        if skill.name in config.exclude_skills:
            return FilterResult(
                skill=skill,
                eligible=False,
                reason=f"Skill '{skill.name}' is in exclude list",
            )

        # 4. Check bundled allowlist
        if skill.source == SkillSource.BUNDLED:
            if config.allow_bundled is not None and skill.name not in config.allow_bundled:
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"Bundled skill '{skill.name}' not in allowlist",
                )

        # 5. Check OS requirements
        requires = skill.metadata.requires
        if requires.os:
            platform = context.platform or sys.platform
            if platform not in requires.os:
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"Skill requires OS {requires.os}, current is {platform}",
                )

        # 6. Check required binaries (ALL must exist)
        for bin_name in requires.bins:
            if not self._has_binary(bin_name, context):
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"Required binary '{bin_name}' not found",
                )

        # 7. Check any-of binaries (at least ONE must exist)
        if requires.any_bins:
            if not any(self._has_binary(b, context) for b in requires.any_bins):
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"None of required binaries found: {requires.any_bins}",
                )

        # 8. Check required environment variables
        for env_name in requires.env:
            if not self._has_env(env_name, skill, skill_config, context):
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"Required env var '{env_name}' not set",
                )

        # 9. Check required config paths
        for config_path in requires.config:
            if not self._has_config(config_path, context):
                return FilterResult(
                    skill=skill,
                    eligible=False,
                    reason=f"Required config '{config_path}' not set",
                )

        return FilterResult(skill=skill, eligible=True)

    def _has_binary(self, name: str, context: FilterContext) -> bool:
        """Check if a binary is available."""
        # Check context first (for remote/cached availability)
        if name in context.available_bins:
            return True
        # Fall back to PATH lookup
        return shutil.which(name) is not None

    def _has_env(
        self,
        name: str,
        skill: Skill,
        skill_config: SkillEntryConfig,
        context: FilterContext,
    ) -> bool:
        """Check if an environment variable is available."""
        # Check context
        if name in context.env_vars:
            return True
        # Check skill config overrides
        if name in skill_config.env:
            return True
        # Check if this is the primary env and we have an API key
        if skill.metadata.primary_env == name and skill_config.api_key:
            return True
        # Check actual environment
        return name in os.environ

    def _has_config(self, path: str, context: FilterContext) -> bool:
        """Check if a config path is truthy."""
        return context.config_values.get(path, False)
