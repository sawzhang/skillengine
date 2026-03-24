"""Grep tool - search file contents with regex."""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.grep")

# Default limit on number of matching files/lines returned
_DEFAULT_LIMIT = 50

# Maximum number of characters in grep output
_MAX_OUTPUT = 100_000


class GrepTool(BaseTool):
    """Search file contents using regex patterns."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a regex pattern. Uses ripgrep (rg) when available "
            "for speed, falling back to Python regex. Returns matching lines with file "
            "paths and line numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search in. Defaults to current working directory."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}'). "
                        "Only files matching this pattern will be searched."
                    ),
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Perform case-insensitive matching. Defaults to false.",
                    "default": False,
                },
                "context_lines": {
                    "type": "integer",
                    "description": (
                        "Number of context lines to show before and after each match. "
                        "Defaults to 0."
                    ),
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of results to return. Defaults to {_DEFAULT_LIMIT}."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        pattern = args.get("pattern", "")
        path = args.get("path", "")
        glob_filter = args.get("glob", "")
        case_insensitive = args.get("case_insensitive", False)
        context_lines = args.get("context_lines", 0)
        limit = args.get("limit", _DEFAULT_LIMIT)

        if not pattern:
            return "Error: pattern is required."

        search_path = self._resolve_path(path) if path else Path(self.cwd)

        if not search_path.exists():
            return f"Error: path not found: {search_path}"

        # Try ripgrep first for speed
        rg_path = shutil.which("rg")
        if rg_path:
            return await self._search_ripgrep(
                rg_path,
                pattern,
                search_path,
                glob_filter,
                case_insensitive,
                context_lines,
                limit,
            )

        # Fall back to Python regex
        return self._search_python(
            pattern,
            search_path,
            glob_filter,
            case_insensitive,
            context_lines,
            limit,
        )

    async def _search_ripgrep(
        self,
        rg_path: str,
        pattern: str,
        search_path: Path,
        glob_filter: str,
        case_insensitive: bool,
        context_lines: int,
        limit: int,
    ) -> str:
        """Search using ripgrep subprocess."""
        cmd = [rg_path, "--line-number", "--no-heading", "--color=never"]

        if case_insensitive:
            cmd.append("--case-sensitive=false")
            cmd.append("-i")

        if context_lines > 0:
            cmd.append(f"-C{context_lines}")

        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        cmd.extend(["--max-count", str(limit)])
        cmd.append(pattern)
        cmd.append(str(search_path))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            return "Error: search timed out after 30s."
        except Exception as e:
            logger.warning("ripgrep failed, falling back to Python: %s", e)
            return self._search_python(
                pattern,
                search_path,
                glob_filter,
                case_insensitive,
                context_lines,
                limit,
            )

        if process.returncode == 1:
            # rg returns 1 when no matches found
            return "No matches found."

        if process.returncode not in (0, 1) and process.returncode is not None:
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            if stderr_str:
                return f"Error: {stderr_str}"
            return f"Error: ripgrep exited with code {process.returncode}"

        output = stdout.decode("utf-8", errors="replace").rstrip()
        if not output:
            return "No matches found."

        # Truncate if too long
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n... (output truncated)"

        return output

    def _search_python(
        self,
        pattern: str,
        search_path: Path,
        glob_filter: str,
        case_insensitive: bool,
        context_lines: int,
        limit: int,
    ) -> str:
        """Fallback: search using Python regex."""
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"

        results: list[str] = []
        match_count = 0

        # Collect files to search
        files = self._collect_files(search_path, glob_filter)

        for file_path in files:
            if match_count >= limit:
                break

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = text.splitlines()
            file_matches: list[tuple[int, str]] = []

            for line_no, line in enumerate(lines, 1):
                if compiled.search(line):
                    file_matches.append((line_no, line))

            if not file_matches:
                continue

            # Format matches with context
            rel_path = self._relative_display(file_path)
            for line_no, line in file_matches:
                if match_count >= limit:
                    break

                if context_lines > 0:
                    # Add context lines
                    start = max(0, line_no - 1 - context_lines)
                    end = min(len(lines), line_no + context_lines)
                    for ctx_no in range(start, end):
                        prefix = ">" if ctx_no == line_no - 1 else " "
                        results.append(f"{rel_path}:{ctx_no + 1}:{prefix}{lines[ctx_no]}")
                    results.append("--")
                else:
                    results.append(f"{rel_path}:{line_no}:{line}")

                match_count += 1

        if not results:
            return "No matches found."

        output = "\n".join(results)
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n... (output truncated)"

        return output

    def _collect_files(self, search_path: Path, glob_filter: str) -> list[Path]:
        """Collect files to search, respecting glob filter."""
        if search_path.is_file():
            return [search_path]

        if glob_filter:
            return sorted(search_path.rglob(glob_filter))

        # Default: search all files recursively, skipping common non-text dirs
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
        for item in sorted(search_path.rglob("*")):
            # Skip entries inside ignored directories
            if any(part in skip_dirs for part in item.parts):
                continue
            if item.is_file():
                files.append(item)
        return files

    def _relative_display(self, path: Path) -> str:
        """Get a relative display path for output."""
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p
