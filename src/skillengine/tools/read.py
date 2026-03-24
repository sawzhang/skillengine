"""Read tool - read file contents with optional line range."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.read")

# Image extensions that should be returned as base64
_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".ico",
    ".tiff",
    ".tif",
}

# Maximum number of lines to read by default
_DEFAULT_LIMIT = 2000

# Maximum line length before truncation
_MAX_LINE_LENGTH = 2000


class ReadTool(BaseTool):
    """Read file contents with optional line range and image support."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. For text files, returns numbered lines. "
            "For image files, returns base64-encoded content. "
            "Use offset and limit to read specific line ranges in large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": ("Line number to start reading from (1-based). Defaults to 1."),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of lines to read. Defaults to {_DEFAULT_LIMIT}."
                    ),
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        file_path = args.get("file_path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", _DEFAULT_LIMIT)

        if not file_path:
            return "Error: file_path is required."

        resolved = self._resolve_path(file_path)

        if not resolved.exists():
            return f"Error: file not found: {resolved}"

        if resolved.is_dir():
            return f"Error: {resolved} is a directory, not a file. Use the ls tool instead."

        # Check if it's an image file
        ext = resolved.suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            return self._read_image(resolved)

        # Check if it's a binary file
        if self._is_binary(resolved):
            size = resolved.stat().st_size
            mime = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            return f"Binary file: {resolved} ({mime}, {self._format_size(size)})"

        return self._read_text(resolved, offset, limit)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p

    def _read_image(self, path: Path) -> str:
        """Read an image file and return base64-encoded content."""
        try:
            data = path.read_bytes()
            encoded = base64.b64encode(data).decode("ascii")
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            size = len(data)
            return (
                f"Image file: {path.name} ({mime}, {self._format_size(size)})\n"
                f"base64:{mime};{encoded}"
            )
        except Exception as e:
            logger.warning("Failed to read image %s: %s", path, e)
            return f"Error reading image file: {e}"

    def _read_text(self, path: Path, offset: int, limit: int) -> str:
        """Read a text file with line numbers."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            return f"Error reading file: {e}"

        all_lines = text.splitlines(keepends=True)
        total_lines = len(all_lines)

        if total_lines == 0:
            return f"File {path} is empty (0 lines)."

        # Clamp offset to valid range (1-based)
        offset = max(1, min(offset, total_lines))
        # Slice to the requested range (convert to 0-based)
        start_idx = offset - 1
        end_idx = min(start_idx + limit, total_lines)
        selected = all_lines[start_idx:end_idx]

        # Format with line numbers, like `cat -n`
        # Calculate padding width based on the largest line number
        max_line_no = end_idx
        width = len(str(max_line_no))
        output_lines: list[str] = []
        for i, line in enumerate(selected, start=offset):
            # Strip trailing newline for display
            content = line.rstrip("\n").rstrip("\r")
            # Truncate very long lines
            if len(content) > _MAX_LINE_LENGTH:
                content = content[:_MAX_LINE_LENGTH] + "... (truncated)"
            output_lines.append(f"{i:>{width}}\t{content}")

        result = "\n".join(output_lines)

        # Add a note if we didn't read the whole file
        if start_idx > 0 or end_idx < total_lines:
            result += (
                f"\n\n(Showing lines {offset}-{end_idx} of {total_lines} total. "
                f"Use offset/limit to read more.)"
            )

        return result

    def _is_binary(self, path: Path) -> bool:
        """Heuristic check for binary files by reading the first chunk."""
        try:
            chunk = path.read_bytes()[:8192]
            # If there are null bytes in the first chunk, treat as binary
            if b"\x00" in chunk:
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _format_size(size: int) -> str:
        """Format a byte count as a human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
