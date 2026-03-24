"""Package source resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PackageSource:
    """Represents where a package comes from."""

    type: Literal["local", "pypi", "git"] = "local"
    path: str = ""  # For local: filesystem path
    package: str = ""  # For pypi: package name
    url: str = ""  # For git: repository URL
    ref: str = ""  # For git: branch/tag/commit


def parse_source(source_str: str) -> PackageSource:
    """Parse a source string into a PackageSource.

    Supported formats:
    - ``"./path/to/package"`` -> local
    - ``"/absolute/path"`` -> local
    - ``"package-name"`` -> pypi
    - ``"git+https://..."`` -> git
    - ``"git+ssh://..."`` -> git
    """
    if source_str.startswith("git+"):
        url = source_str[4:]
        ref = ""
        if "@" in url:
            url, ref = url.rsplit("@", 1)
        return PackageSource(type="git", url=url, ref=ref)

    if source_str.startswith(("./", "../", "/")):
        return PackageSource(type="local", path=source_str)

    path = Path(source_str)
    if path.exists():
        return PackageSource(type="local", path=source_str)

    return PackageSource(type="pypi", package=source_str)
