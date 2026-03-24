"""Context file (AGENTS.md, CLAUDE.md) discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONTEXT_FILE_NAMES = ["AGENTS.md", "CLAUDE.md"]


@dataclass
class ContextFile:
    """A loaded context file."""

    path: Path
    content: str


def _find_context_file_in_dir(directory: Path) -> ContextFile | None:
    """Check a directory for context files, returning the first found."""
    for name in CONTEXT_FILE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8")
                return ContextFile(path=candidate.resolve(), content=content)
            except (OSError, UnicodeDecodeError):
                continue
    return None


def load_context_files(
    cwd: Path,
    home_dir: Path | None = None,
) -> list[ContextFile]:
    """Load context files from standard locations.

    Search order:
    1. ~/.skillengine/AGENTS.md or CLAUDE.md (global)
    2. Walk from cwd up to root, collecting AGENTS.md or CLAUDE.md from each directory

    Files are deduplicated by resolved path. Closest files appear last (highest priority).
    """
    seen_paths: set[Path] = set()
    files: list[ContextFile] = []

    # 1. Global context file
    agent_dir = (home_dir or Path.home()) / ".skillengine"
    if agent_dir.is_dir():
        ctx = _find_context_file_in_dir(agent_dir)
        if ctx and ctx.path not in seen_paths:
            seen_paths.add(ctx.path)
            files.append(ctx)

    # 2. Walk up from cwd to root
    ancestor_files: list[ContextFile] = []
    current = cwd.resolve()
    while True:
        ctx = _find_context_file_in_dir(current)
        if ctx and ctx.path not in seen_paths:
            seen_paths.add(ctx.path)
            ancestor_files.append(ctx)
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Reverse so closest directory is last (highest priority)
    ancestor_files.reverse()
    files.extend(ancestor_files)

    return files
