"""
Base skill filter interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from skillengine.config import SkillsConfig
from skillengine.models import Skill


@dataclass
class FilterContext:
    """Context for skill filtering decisions."""

    platform: str = ""  # Current platform (linux, darwin, win32)
    available_bins: set[str] = field(default_factory=set)  # Available binaries
    env_vars: set[str] = field(default_factory=set)  # Available env vars
    config_values: dict[str, bool] = field(default_factory=dict)  # Config path -> truthy


@dataclass
class FilterResult:
    """Result of filtering a skill."""

    skill: Skill
    eligible: bool
    reason: str | None = None  # Reason for ineligibility


class SkillFilter(ABC):
    """
    Abstract base class for skill filters.

    Implement this interface to customize skill eligibility logic.
    """

    @abstractmethod
    def filter(
        self,
        skill: Skill,
        config: SkillsConfig,
        context: FilterContext,
    ) -> FilterResult:
        """
        Determine if a skill is eligible.

        Args:
            skill: The skill to check
            config: Skills configuration
            context: Runtime context (platform, available bins, etc.)

        Returns:
            FilterResult indicating eligibility
        """
        pass

    def filter_all(
        self,
        skills: list[Skill],
        config: SkillsConfig,
        context: FilterContext,
    ) -> list[FilterResult]:
        """Filter multiple skills."""
        return [self.filter(skill, config, context) for skill in skills]

    def get_eligible(
        self,
        skills: list[Skill],
        config: SkillsConfig,
        context: FilterContext,
    ) -> list[Skill]:
        """Get only eligible skills."""
        return [
            result.skill for result in self.filter_all(skills, config, context) if result.eligible
        ]
