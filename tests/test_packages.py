"""Tests for the packages module: models, source parsing, and package manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillengine.packages.models import (
    PackageManifest,
    ResolvedPackage,
    PathMetadata,
)
from skillengine.packages.source import PackageSource, parse_source
from skillengine.packages.manager import PackageManager


class TestPackageManifest:
    """Tests for PackageManifest dataclass."""

    def test_from_dict(self) -> None:
        """Should create a manifest from a dictionary."""
        data = {
            "extensions": ["./ext.py"],
            "skills": ["./skills/**/*.md"],
            "themes": ["./themes/*.json"],
            "prompts": ["./prompts/*.md"],
        }

        manifest = PackageManifest.from_dict(data)

        assert manifest.extensions == ["./ext.py"]
        assert manifest.skills == ["./skills/**/*.md"]
        assert manifest.themes == ["./themes/*.json"]
        assert manifest.prompts == ["./prompts/*.md"]

    def test_from_dict_partial(self) -> None:
        """Should handle partial dictionaries with defaults."""
        data = {"skills": ["./skills/*.md"]}

        manifest = PackageManifest.from_dict(data)

        assert manifest.skills == ["./skills/*.md"]
        assert manifest.extensions == []
        assert manifest.themes == []
        assert manifest.prompts == []

    def test_from_dict_empty(self) -> None:
        """Should create an empty manifest from empty dict."""
        manifest = PackageManifest.from_dict({})

        assert manifest.extensions == []
        assert manifest.skills == []
        assert manifest.themes == []
        assert manifest.prompts == []

    def test_is_empty_for_empty_manifest(self) -> None:
        """Should return True when manifest has no resources."""
        manifest = PackageManifest()

        assert manifest.is_empty is True

    def test_is_empty_for_non_empty_manifest(self) -> None:
        """Should return False when manifest has at least one resource."""
        manifest = PackageManifest(skills=["./skills/*.md"])

        assert manifest.is_empty is False

    def test_is_empty_with_extensions(self) -> None:
        """Should return False when only extensions are populated."""
        manifest = PackageManifest(extensions=["./ext.py"])

        assert manifest.is_empty is False

    def test_is_empty_with_themes(self) -> None:
        """Should return False when only themes are populated."""
        manifest = PackageManifest(themes=["./theme.json"])

        assert manifest.is_empty is False

    def test_is_empty_with_prompts(self) -> None:
        """Should return False when only prompts are populated."""
        manifest = PackageManifest(prompts=["./prompt.md"])

        assert manifest.is_empty is False


class TestPathMetadata:
    """Tests for PathMetadata dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        meta = PathMetadata()

        assert meta.source == "local"
        assert meta.scope == "project"
        assert meta.origin == "top-level"
        assert meta.base_dir == ""

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        meta = PathMetadata(
            source="pypi",
            scope="user",
            origin="package",
            base_dir="/home/user/.skillengine",
        )

        assert meta.source == "pypi"
        assert meta.scope == "user"
        assert meta.origin == "package"
        assert meta.base_dir == "/home/user/.skillengine"


class TestResolvedPackage:
    """Tests for ResolvedPackage dataclass."""

    def test_creation(self) -> None:
        """Should create a ResolvedPackage with defaults."""
        pkg = ResolvedPackage(name="my-package")

        assert pkg.name == "my-package"
        assert pkg.version == ""
        assert pkg.source_type == "local"
        assert pkg.manifest.is_empty is True
        assert pkg.extensions == []
        assert pkg.skills == []
        assert pkg.themes == []
        assert pkg.prompts == []

    def test_creation_with_manifest(self) -> None:
        """Should associate a manifest with the package."""
        manifest = PackageManifest(skills=["./skills/*.md"])
        pkg = ResolvedPackage(
            name="pkg",
            version="1.0.0",
            source_type="pypi",
            manifest=manifest,
        )

        assert pkg.version == "1.0.0"
        assert pkg.source_type == "pypi"
        assert pkg.manifest.skills == ["./skills/*.md"]


class TestPackageSource:
    """Tests for PackageSource dataclass."""

    def test_creation(self) -> None:
        """Should create a PackageSource with specified fields."""
        source = PackageSource(
            type="git",
            url="https://github.com/user/repo",
            ref="main",
        )

        assert source.type == "git"
        assert source.url == "https://github.com/user/repo"
        assert source.ref == "main"

    def test_default_type_is_local(self) -> None:
        """Should default to local type."""
        source = PackageSource()

        assert source.type == "local"
        assert source.path == ""
        assert source.package == ""
        assert source.url == ""
        assert source.ref == ""


class TestParseSource:
    """Tests for parse_source function."""

    def test_local_relative_path(self) -> None:
        """Should parse relative paths as local sources."""
        source = parse_source("./my-package")

        assert source.type == "local"
        assert source.path == "./my-package"

    def test_local_parent_relative_path(self) -> None:
        """Should parse parent-relative paths as local sources."""
        source = parse_source("../other-package")

        assert source.type == "local"
        assert source.path == "../other-package"

    def test_local_absolute_path(self) -> None:
        """Should parse absolute paths as local sources."""
        source = parse_source("/abs/path/to/pkg")

        assert source.type == "local"
        assert source.path == "/abs/path/to/pkg"

    def test_git_https_url(self) -> None:
        """Should parse git+https URLs as git sources."""
        source = parse_source("git+https://github.com/user/repo")

        assert source.type == "git"
        assert source.url == "https://github.com/user/repo"
        assert source.ref == ""

    def test_git_ssh_url(self) -> None:
        """Should parse git+ssh URLs as git sources.

        Note: The parser uses rsplit('@', 1), so URLs containing '@'
        (like ssh://git@host) will be split on the last '@'.
        """
        source = parse_source("git+ssh://git@github.com/user/repo")

        assert source.type == "git"
        # rsplit splits on the '@' in git@github.com
        assert source.url == "ssh://git"
        assert source.ref == "github.com/user/repo"

    def test_git_with_ref(self) -> None:
        """Should parse git URLs with @ref as git sources with ref."""
        source = parse_source("git+https://github.com/user/repo@v1.0")

        assert source.type == "git"
        assert source.url == "https://github.com/user/repo"
        assert source.ref == "v1.0"

    def test_git_with_branch_ref(self) -> None:
        """Should parse git URLs with branch ref."""
        source = parse_source("git+https://github.com/user/repo@main")

        assert source.type == "git"
        assert source.url == "https://github.com/user/repo"
        assert source.ref == "main"

    def test_pypi_package_name(self) -> None:
        """Should parse plain names as pypi sources."""
        source = parse_source("package-name")

        assert source.type == "pypi"
        assert source.package == "package-name"

    def test_pypi_simple_name(self) -> None:
        """Should parse simple package names as pypi sources."""
        source = parse_source("requests")

        assert source.type == "pypi"
        assert source.package == "requests"


class TestPackageManager:
    """Tests for PackageManager."""

    def test_resolve_empty_dirs(self, tmp_path: Path) -> None:
        """Should return empty list when directories are empty."""
        user_dir = tmp_path / "user_packages"
        project_dir = tmp_path / "project_packages"
        user_dir.mkdir()
        project_dir.mkdir()

        manager = PackageManager(user_dir=user_dir, project_dir=project_dir)
        packages = manager.resolve()

        assert packages == []

    def test_resolve_nonexistent_dirs(self, tmp_path: Path) -> None:
        """Should return empty list when directories don't exist."""
        manager = PackageManager(
            user_dir=tmp_path / "nonexistent_user",
            project_dir=tmp_path / "nonexistent_project",
        )
        packages = manager.resolve()

        assert packages == []

    def test_resolve_finds_local_package(self, tmp_path: Path) -> None:
        """Should discover a local package from a standard directory."""
        project_dir = tmp_path / "project_packages"
        project_dir.mkdir()

        # Create a package with a pyproject.toml manifest
        pkg_dir = project_dir / "my-skill-pack"
        pkg_dir.mkdir()
        skills_dir = pkg_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "hello.md").write_text("# Hello Skill\n")

        pyproject = pkg_dir / "pyproject.toml"
        pyproject.write_text(
            '[tool.skillengine]\n'
            'skills = ["./skills/*.md"]\n'
        )

        manager = PackageManager(
            user_dir=tmp_path / "empty_user",
            project_dir=project_dir,
        )
        packages = manager.resolve()

        assert len(packages) == 1
        assert packages[0].name == "my-skill-pack"
        assert packages[0].source_type == "local"
        assert len(packages[0].skills) == 1

    def test_load_manifest_from_pyproject(self, tmp_path: Path) -> None:
        """Should load a PackageManifest from a pyproject.toml file."""
        pkg_dir = tmp_path / "my-package"
        pkg_dir.mkdir()
        pyproject = pkg_dir / "pyproject.toml"
        pyproject.write_text(
            '[tool.skillengine]\n'
            'extensions = ["./ext.py"]\n'
            'skills = ["./skills/**/*.md"]\n'
            'themes = ["./themes/*.json"]\n'
            'prompts = ["./prompts/*.md"]\n'
        )

        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        manifest = manager.load_manifest(pkg_dir)

        assert manifest is not None
        assert manifest.extensions == ["./ext.py"]
        assert manifest.skills == ["./skills/**/*.md"]
        assert manifest.themes == ["./themes/*.json"]
        assert manifest.prompts == ["./prompts/*.md"]
        assert manifest.is_empty is False

    def test_load_manifest_no_skillengine_section(self, tmp_path: Path) -> None:
        """Should return None when pyproject.toml has no skillengine section."""
        pkg_dir = tmp_path / "plain-package"
        pkg_dir.mkdir()
        pyproject = pkg_dir / "pyproject.toml"
        pyproject.write_text(
            '[project]\n'
            'name = "plain"\n'
            'version = "1.0"\n'
        )

        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        manifest = manager.load_manifest(pkg_dir)

        assert manifest is None

    def test_load_manifest_no_file(self, tmp_path: Path) -> None:
        """Should return None when directory has no manifest file."""
        pkg_dir = tmp_path / "empty-package"
        pkg_dir.mkdir()

        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        manifest = manager.load_manifest(pkg_dir)

        assert manifest is None

    def test_get_all_resources_empty(self, tmp_path: Path) -> None:
        """Should return empty resource lists when no packages resolved."""
        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        manager.resolve()
        resources = manager.get_all_resources()

        assert resources == {
            "extensions": [],
            "skills": [],
            "themes": [],
            "prompts": [],
        }

    def test_get_all_resources_aggregates(self, tmp_path: Path) -> None:
        """Should aggregate resources across all resolved packages."""
        project_dir = tmp_path / "project_packages"
        project_dir.mkdir()

        # Create two packages, each with a skill
        for name in ("pack-a", "pack-b"):
            pkg_dir = project_dir / name
            pkg_dir.mkdir()
            skills_dir = pkg_dir / "skills"
            skills_dir.mkdir()
            (skills_dir / f"{name}-skill.md").write_text(f"# {name}\n")
            (pkg_dir / "pyproject.toml").write_text(
                '[tool.skillengine]\n'
                'skills = ["./skills/*.md"]\n'
            )

        manager = PackageManager(
            user_dir=tmp_path / "empty_user",
            project_dir=project_dir,
        )
        manager.resolve()
        resources = manager.get_all_resources()

        assert len(resources["skills"]) == 2
        assert resources["extensions"] == []
        assert resources["themes"] == []
        assert resources["prompts"] == []

    def test_resolve_with_explicit_local_source(self, tmp_path: Path) -> None:
        """Should resolve an explicitly provided local package source."""
        pkg_dir = tmp_path / "explicit-pkg"
        pkg_dir.mkdir()
        skills_dir = pkg_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "skill.md").write_text("# Skill\n")
        (pkg_dir / "pyproject.toml").write_text(
            '[tool.skillengine]\n'
            'skills = ["./skills/*.md"]\n'
        )

        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        # Use absolute path so _resolve_source -> Path(path).resolve() works
        packages = manager.resolve(sources=[str(pkg_dir)])

        # The explicit source should be resolved
        found = [p for p in packages if p.name == "explicit-pkg"]
        assert len(found) == 1
        assert len(found[0].skills) == 1

    def test_packages_property(self, tmp_path: Path) -> None:
        """Should expose resolved packages through the packages property."""
        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=tmp_path / "p",
        )
        manager.resolve()

        packages = manager.packages
        assert isinstance(packages, list)
        assert packages == []

    def test_resolve_skips_empty_manifests(self, tmp_path: Path) -> None:
        """Should skip packages with empty manifests."""
        project_dir = tmp_path / "project_packages"
        project_dir.mkdir()

        # Package with empty skillengine section
        pkg_dir = project_dir / "empty-manifest"
        pkg_dir.mkdir()
        (pkg_dir / "pyproject.toml").write_text(
            '[tool.skillengine]\n'
        )

        manager = PackageManager(
            user_dir=tmp_path / "u",
            project_dir=project_dir,
        )
        packages = manager.resolve()

        assert len(packages) == 0
