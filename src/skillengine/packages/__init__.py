"""Package management for skills, extensions, themes, and prompts."""

from __future__ import annotations

from skillengine.packages.manager import PackageManager
from skillengine.packages.models import (
    PackageManifest,
    PathMetadata,
    ResolvedPackage,
    ResolvedResource,
)
from skillengine.packages.source import PackageSource, parse_source

__all__ = [
    "PackageManifest",
    "ResolvedPackage",
    "ResolvedResource",
    "PathMetadata",
    "PackageSource",
    "parse_source",
    "PackageManager",
]
