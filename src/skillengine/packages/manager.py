"""Package manager for resolving and loading packages."""

from __future__ import annotations

import glob as glob_module
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from skillengine.packages.models import (
    PackageManifest,
    PathMetadata,
    ResolvedPackage,
    ResolvedResource,
)
from skillengine.packages.source import PackageSource, parse_source


class PackageManager:
    """Manages discovery, resolution, and loading of skill packages.

    Package sources:
    - User scope: ``~/.skillengine/packages/``
    - Project scope: ``.skillengine/packages/``
    - ``pyproject.toml`` ``[tool.skillengine]`` section
    """

    def __init__(
        self,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._user_dir = user_dir or (Path.home() / ".skillengine" / "packages")
        self._project_dir = project_dir or (Path.cwd() / ".skillengine" / "packages")
        self._packages: list[ResolvedPackage] = []

    @property
    def packages(self) -> list[ResolvedPackage]:
        """Get all resolved packages."""
        return list(self._packages)

    def resolve(
        self,
        sources: list[str | PackageSource] | None = None,
    ) -> list[ResolvedPackage]:
        """Resolve packages from sources and standard directories.

        Args:
            sources: Explicit package sources to resolve.
                If ``None``, auto-discover from standard directories.
        """
        self._packages = []

        # Auto-discover from standard directories
        for scope_dir, scope in [
            (self._user_dir, "user"),
            (self._project_dir, "project"),
        ]:
            if scope_dir.is_dir():
                for item in sorted(scope_dir.iterdir()):
                    if item.is_dir():
                        pkg = self._resolve_local(item, scope=scope)
                        if pkg:
                            self._packages.append(pkg)

        # Resolve from pyproject.toml in current directory
        pyproject_path = Path.cwd() / "pyproject.toml"
        if pyproject_path.is_file():
            pkg = self._resolve_pyproject(pyproject_path)
            if pkg:
                self._packages.append(pkg)

        # Resolve explicit sources
        if sources:
            for src in sources:
                if isinstance(src, str):
                    src = parse_source(src)
                pkg = self._resolve_source(src)
                if pkg:
                    self._packages.append(pkg)

        return self._packages

    def _resolve_source(
        self,
        source: PackageSource,
    ) -> ResolvedPackage | None:
        """Resolve a single package source."""
        if source.type == "local":
            path = Path(source.path).resolve()
            if path.is_dir():
                return self._resolve_local(path, scope="temporary")
        return None

    def _resolve_local(
        self,
        path: Path,
        scope: str = "project",
    ) -> ResolvedPackage | None:
        """Resolve a local package directory."""
        manifest = self.load_manifest(path)
        if manifest is None or manifest.is_empty:
            return None

        metadata = PathMetadata(
            source="local",
            scope=scope,
            origin="package",
            base_dir=str(path),
        )

        return ResolvedPackage(
            name=path.name,
            source_type="local",
            base_dir=path,
            manifest=manifest,
            extensions=self._resolve_globs(
                path,
                manifest.extensions,
                metadata,
            ),
            skills=self._resolve_globs(
                path,
                manifest.skills,
                metadata,
            ),
            themes=self._resolve_globs(
                path,
                manifest.themes,
                metadata,
            ),
            prompts=self._resolve_globs(
                path,
                manifest.prompts,
                metadata,
            ),
        )

    def _resolve_pyproject(
        self,
        pyproject_path: Path,
    ) -> ResolvedPackage | None:
        """Resolve package from pyproject.toml."""
        if tomllib is None:
            return None

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return None

        sk_config = data.get("tool", {}).get("skillengine", {})
        if not sk_config:
            return None

        manifest = PackageManifest.from_dict(sk_config)
        if manifest.is_empty:
            return None

        base_dir = pyproject_path.parent
        metadata = PathMetadata(
            source="local",
            scope="project",
            origin="top-level",
            base_dir=str(base_dir),
        )

        project_name = data.get("project", {}).get("name", base_dir.name)
        project_version = data.get("project", {}).get("version", "")

        return ResolvedPackage(
            name=project_name,
            version=project_version,
            source_type="local",
            base_dir=base_dir,
            manifest=manifest,
            extensions=self._resolve_globs(
                base_dir,
                manifest.extensions,
                metadata,
            ),
            skills=self._resolve_globs(
                base_dir,
                manifest.skills,
                metadata,
            ),
            themes=self._resolve_globs(
                base_dir,
                manifest.themes,
                metadata,
            ),
            prompts=self._resolve_globs(
                base_dir,
                manifest.prompts,
                metadata,
            ),
        )

    def _resolve_globs(
        self,
        base_dir: Path,
        patterns: list[str],
        metadata: PathMetadata,
    ) -> list[ResolvedResource]:
        """Resolve glob patterns relative to *base_dir*."""
        resources: list[ResolvedResource] = []
        seen: set[Path] = set()

        for pattern in patterns:
            full_pattern = str(base_dir / pattern)
            for match_str in sorted(
                glob_module.glob(full_pattern, recursive=True),
            ):
                match_path = Path(match_str).resolve()
                if match_path not in seen and match_path.is_file():
                    seen.add(match_path)
                    resources.append(
                        ResolvedResource(
                            path=match_path,
                            metadata=metadata,
                        ),
                    )

        return resources

    def load_manifest(self, path: Path) -> PackageManifest | None:
        """Load a package manifest from a directory.

        Looks for:
        1. ``pyproject.toml`` with ``[tool.skillengine]`` section
        2. ``package.yaml`` with manifest fields
        """
        # Try pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.is_file() and tomllib is not None:
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                sk_config = data.get("tool", {}).get("skillengine", {})
                if sk_config:
                    return PackageManifest.from_dict(sk_config)
            except Exception:
                pass

        # Try package.yaml
        package_yaml = path / "package.yaml"
        if package_yaml.is_file():
            try:
                import yaml

                with open(package_yaml) as f:
                    data = yaml.safe_load(f)
                if data:
                    return PackageManifest.from_dict(data)
            except Exception:
                pass

        return None

    def get_all_resources(
        self,
    ) -> dict[str, list[ResolvedResource]]:
        """Get all resolved resources across all packages."""
        result: dict[str, list[ResolvedResource]] = {
            "extensions": [],
            "skills": [],
            "themes": [],
            "prompts": [],
        }
        for pkg in self._packages:
            result["extensions"].extend(pkg.extensions)
            result["skills"].extend(pkg.skills)
            result["themes"].extend(pkg.themes)
            result["prompts"].extend(pkg.prompts)
        return result
