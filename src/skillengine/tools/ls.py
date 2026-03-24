"""Ls tool - list directory contents."""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.ls")


class LsTool(BaseTool):
    """List directory contents with optional detailed output."""

    @property
    def name(self) -> str:
        return "ls"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Shows files and subdirectories. "
            "Supports recursive listing and detailed (long) format with permissions, "
            "sizes, and dates."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path to list. Defaults to current working directory."
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List contents recursively. Defaults to false.",
                    "default": False,
                },
                "long_format": {
                    "type": "boolean",
                    "description": (
                        "Show detailed information (permissions, size, date). Defaults to false."
                    ),
                    "default": False,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (dotfiles). Defaults to false.",
                    "default": False,
                },
            },
        }

    async def execute(self, args: dict[str, Any]) -> str:
        path = args.get("path", "")
        recursive = args.get("recursive", False)
        long_format = args.get("long_format", False)
        include_hidden = args.get("include_hidden", False)

        target = self._resolve_path(path) if path else Path(self.cwd)

        if not target.exists():
            return f"Error: path not found: {target}"

        if not target.is_dir():
            # If it's a file, show info about that file
            if long_format:
                return self._format_entry_long(target)
            return str(target.name)

        if recursive:
            return self._list_recursive(target, long_format, include_hidden)

        return self._list_directory(target, long_format, include_hidden)

    def _list_directory(
        self,
        directory: Path,
        long_format: bool,
        include_hidden: bool,
    ) -> str:
        """List a single directory's contents."""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            return f"Error: permission denied: {directory}"
        except OSError as e:
            return f"Error listing directory: {e}"

        if not include_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        if not entries:
            return f"{directory}: (empty)"

        lines: list[str] = []
        for entry in entries:
            if long_format:
                lines.append(self._format_entry_long(entry))
            else:
                suffix = "/" if entry.is_dir() else ""
                lines.append(f"{entry.name}{suffix}")

        return "\n".join(lines)

    def _list_recursive(
        self,
        directory: Path,
        long_format: bool,
        include_hidden: bool,
    ) -> str:
        """List directory contents recursively."""
        skip_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
        }

        lines: list[str] = []
        max_entries = 1000  # Safety limit for recursive listing
        count = 0

        for root_path, dirs, files in os.walk(directory):
            root = Path(root_path)

            # Skip ignored directories
            dirs[:] = sorted(
                d for d in dirs if d not in skip_dirs and (include_hidden or not d.startswith("."))
            )

            # Filter hidden files
            if not include_hidden:
                files = [f for f in files if not f.startswith(".")]

            files = sorted(files)

            # Show the directory header
            try:
                rel = root.relative_to(directory)
                dir_display = str(rel) if str(rel) != "." else "."
            except ValueError:
                dir_display = str(root)

            if lines:  # Add blank line between directories
                lines.append("")
            lines.append(f"{dir_display}/:")

            entries = []
            for d in sorted(dirs):
                entries.append((root / d, True))
            for f in files:
                entries.append((root / f, False))

            for entry_path, is_dir in entries:
                count += 1
                if count > max_entries:
                    lines.append(
                        f"\n... (stopped after {max_entries} entries, use a more specific path)"
                    )
                    return "\n".join(lines)

                if long_format:
                    lines.append(f"  {self._format_entry_long(entry_path)}")
                else:
                    suffix = "/" if is_dir else ""
                    lines.append(f"  {entry_path.name}{suffix}")

        if not lines:
            return f"{directory}: (empty)"

        return "\n".join(lines)

    def _format_entry_long(self, path: Path) -> str:
        """Format a single entry in long format (like ls -l)."""
        try:
            st = path.stat()
        except OSError:
            return f"? ? ? {path.name}"

        # Permissions string
        mode = st.st_mode
        perms = self._format_permissions(mode)

        # Type indicator
        if stat.S_ISDIR(mode):
            type_char = "d"
        elif stat.S_ISLNK(mode):
            type_char = "l"
        else:
            type_char = "-"

        # Size
        size = self._format_size(st.st_size)

        # Modification time
        mtime = datetime.fromtimestamp(st.st_mtime)
        now = datetime.now()
        if mtime.year == now.year:
            date_str = mtime.strftime("%b %d %H:%M")
        else:
            date_str = mtime.strftime("%b %d  %Y")

        suffix = "/" if stat.S_ISDIR(mode) else ""
        return f"{type_char}{perms}  {size:>8s}  {date_str}  {path.name}{suffix}"

    @staticmethod
    def _format_permissions(mode: int) -> str:
        """Format file permissions as rwxrwxrwx string."""
        perms = ""
        for who in (
            stat.S_IRUSR,
            stat.S_IWUSR,
            stat.S_IXUSR,
            stat.S_IRGRP,
            stat.S_IWGRP,
            stat.S_IXGRP,
            stat.S_IROTH,
            stat.S_IWOTH,
            stat.S_IXOTH,
        ):
            if mode & who:
                # Determine which character based on position
                idx = (
                    stat.S_IRUSR,
                    stat.S_IWUSR,
                    stat.S_IXUSR,
                    stat.S_IRGRP,
                    stat.S_IWGRP,
                    stat.S_IXGRP,
                    stat.S_IROTH,
                    stat.S_IWOTH,
                    stat.S_IXOTH,
                ).index(who)
                chars = "rwxrwxrwx"
                perms += chars[idx]
            else:
                perms += "-"
        return perms

    @staticmethod
    def _format_size(size: int) -> str:
        """Format a byte count as a human-readable string."""
        if size < 1024:
            return f"{size}B"
        for unit in ("KB", "MB", "GB"):
            size_f = size / 1024
            if size_f < 1024 or unit == "GB":
                return f"{size_f:.1f}{unit}"
            size = int(size_f)
        return f"{size}TB"

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p
