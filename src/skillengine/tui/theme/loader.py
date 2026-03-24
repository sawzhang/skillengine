"""Theme loading and discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillengine.tui.theme.models import ThemeInfo


def load_theme(path: Path) -> ThemeInfo:
    """Load a theme from a JSON file.

    Supports variable resolution: a color value like ``"primary"`` is resolved
    against the ``variables`` dict first, then the ``colors`` dict.
    """
    with open(path) as f:
        data: dict[str, Any] = json.load(f)

    name = data.get("name", path.stem)
    description = data.get("description", "")
    author = data.get("author", "")
    variables = data.get("variables", {})
    raw_colors = data.get("colors", {})

    # Resolve variable references
    resolved_colors: dict[str, str] = {}
    for key, value in raw_colors.items():
        if isinstance(value, str) and not value.startswith("#") and value in variables:
            resolved_colors[key] = variables[value]
        elif isinstance(value, str) and not value.startswith("#") and value in raw_colors:
            # Forward reference to another color key
            ref_val = raw_colors[value]
            if isinstance(ref_val, str) and ref_val in variables:
                resolved_colors[key] = variables[ref_val]
            else:
                resolved_colors[key] = str(ref_val)
        else:
            resolved_colors[key] = str(value) if value else ""

    return ThemeInfo(
        name=name,
        description=description,
        author=author,
        colors=resolved_colors,
    )


def discover_themes(
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> list[Path]:
    """Discover theme files from standard directories.

    Search paths:
    - ``~/.skillengine/themes/``
    - ``.skillengine/themes/``
    """
    theme_files: list[Path] = []
    dirs = [
        user_dir or (Path.home() / ".skillengine" / "themes"),
        project_dir or (Path.cwd() / ".skillengine" / "themes"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                if f.is_file():
                    theme_files.append(f)
    return theme_files
