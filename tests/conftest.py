"""Shared pytest fixtures for skillengine tests."""

from pathlib import Path
from textwrap import dedent

import pytest

from skillengine import SkillsConfig, SkillsEngine
from skillengine.filters.base import FilterContext
from skillengine.models import Skill, SkillMetadata, SkillRequirements, SkillSource


@pytest.fixture
def sample_skill() -> Skill:
    """Create a sample skill for testing."""
    return Skill(
        name="test-skill",
        description="A test skill for unit testing",
        content="# Test Skill\n\nThis is a test skill.",
        file_path=Path("/tmp/skills/test-skill/SKILL.md"),
        base_dir=Path("/tmp/skills/test-skill"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(
            emoji="🧪",
            primary_env="TEST_API_KEY",
        ),
    )


@pytest.fixture
def skill_with_requirements() -> Skill:
    """Create a skill with requirements for testing."""
    return Skill(
        name="git-skill",
        description="Git operations",
        content="# Git Skill",
        file_path=Path("/tmp/skills/git-skill/SKILL.md"),
        base_dir=Path("/tmp/skills/git-skill"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(
            emoji="📦",
            requires=SkillRequirements(
                bins=["git"],
                env=["GITHUB_TOKEN"],
                os=["linux", "darwin"],
            ),
        ),
    )


@pytest.fixture
def bundled_skill() -> Skill:
    """Create a bundled skill for testing."""
    return Skill(
        name="bundled-skill",
        description="A bundled skill",
        content="# Bundled Skill",
        file_path=Path("/usr/share/skills/bundled-skill/SKILL.md"),
        base_dir=Path("/usr/share/skills/bundled-skill"),
        source=SkillSource.BUNDLED,
        metadata=SkillMetadata(emoji="📦"),
    )


@pytest.fixture
def always_skill() -> Skill:
    """Create a skill with always=True."""
    return Skill(
        name="always-skill",
        description="Always included skill",
        content="# Always Skill",
        file_path=Path("/tmp/skills/always-skill/SKILL.md"),
        base_dir=Path("/tmp/skills/always-skill"),
        source=SkillSource.WORKSPACE,
        metadata=SkillMetadata(always=True),
    )


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Create a temporary skill directory with sample skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create a simple skill
    simple_dir = skills_dir / "simple-skill"
    simple_dir.mkdir()
    (simple_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: simple-skill
        description: "A simple test skill"
        metadata:
          emoji: "🔧"
        ---
        # Simple Skill

        This is a simple skill for testing.
    """).strip()
    )

    # Create a skill with requirements
    req_dir = skills_dir / "requires-git"
    req_dir.mkdir()
    (req_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: requires-git
        description: "Requires git binary"
        metadata:
          requires:
            bins:
              - git
        ---
        # Git Skill
    """).strip()
    )

    # Create a skill with env requirement
    env_dir = skills_dir / "requires-env"
    env_dir.mkdir()
    (env_dir / "SKILL.md").write_text(
        dedent("""
        ---
        name: requires-env
        description: "Requires API key"
        metadata:
          primary_env: "MY_API_KEY"
          requires:
            env:
              - MY_API_KEY
        ---
        # Env Skill
    """).strip()
    )

    return skills_dir


@pytest.fixture
def engine_with_skills(skill_dir: Path) -> SkillsEngine:
    """Create a SkillsEngine with the test skill directory."""
    config = SkillsConfig(skill_dirs=[skill_dir])
    return SkillsEngine(config=config)


@pytest.fixture
def default_context() -> FilterContext:
    """Create a default filter context."""
    return FilterContext(
        platform="darwin",
        available_bins=set(),
        env_vars=set(),
        config_values={},
    )


@pytest.fixture
def linux_context() -> FilterContext:
    """Create a Linux filter context."""
    return FilterContext(
        platform="linux",
        available_bins={"git", "docker"},
        env_vars={"HOME", "PATH"},
        config_values={},
    )
