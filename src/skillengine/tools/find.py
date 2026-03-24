"""Find tool - find files by glob pattern."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.find")

# Default maximum number of results
_DEFAULT_LIMIT = 200


class FindTool(BaseTool):
    """Find files by glob pattern, respecting .gitignore when possible."""

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). "
            "Results are sorted by modification time (most recent first). "
            "Respects .gitignore when inside a git repository."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match files against "
                        "(e.g. '**/*.py', 'src/**/*.ts', '*.json')."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search in. Defaults to current working directory."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (f"Maximum number of results. Defaults to {_DEFAULT_LIMIT}."),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        pattern = args.get("pattern", "")
        path = args.get("path", "")
        limit = args.get("limit", _DEFAULT_LIMIT)

        if not pattern:
            return "Error: pattern is required."

        search_dir = self._resolve_path(path) if path else Path(self.cwd)

        if not search_dir.exists():
            return f"Error: directory not found: {search_dir}"

        if not search_dir.is_dir():
            return f"Error: {search_dir} is not a directory."

        # Try git ls-files first to respect .gitignore
        git_files = await self._git_ls_files(search_dir, pattern)
        if git_files is not None:
            files = git_files
        else:
            files = self._glob_files(search_dir, pattern)

        if not files:
            return "No files found."

        # Sort by modification time, most recent first
        files_with_mtime: list[tuple[Path, float]] = []
        for f in files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            files_with_mtime.append((f, mtime))

        files_with_mtime.sort(key=lambda x: x[1], reverse=True)

        total = len(files_with_mtime)
        limited = files_with_mtime[:limit]

        lines: list[str] = []
        for f, _mtime in limited:
            lines.append(self._relative_display(f, search_dir))

        result = "\n".join(lines)
        if total > limit:
            result += f"\n\n({total} total files, showing first {limit})"
        else:
            result += f"\n\n({total} files)"

        return result

    async def _git_ls_files(
        self,
        search_dir: Path,
        pattern: str,
    ) -> list[Path] | None:
        """
        Use `git ls-files` to find files respecting .gitignore.

        Returns None if git is not available or the directory is not a git repo.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                cwd=str(search_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0,
            )
        except (FileNotFoundError, asyncio.TimeoutError):
            return None
        except Exception:
            return None

        if process.returncode != 0:
            return None

        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return []

        # Filter files by the glob pattern using pathlib matching
        all_files = output.splitlines()
        matched: list[Path] = []
        for rel in all_files:
            rel_path = Path(rel)
            if rel_path.match(pattern):
                full_path = search_dir / rel_path
                if full_path.is_file():
                    matched.append(full_path)

        return matched

    def _glob_files(self, search_dir: Path, pattern: str) -> list[Path]:
        """Find files using pathlib glob, skipping common ignored directories."""
        skip_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "coverage",
        }

        files: list[Path] = []
        try:
            for item in search_dir.glob(pattern):
                if any(part in skip_dirs for part in item.parts):
                    continue
                if item.is_file():
                    files.append(item)
        except Exception as e:
            logger.warning("Glob error for pattern '%s': %s", pattern, e)

        return files

    def _relative_display(self, path: Path, base: Path) -> str:
        """Get a relative display path for output."""
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p
