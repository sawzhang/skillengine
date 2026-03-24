"""Write tool - create or overwrite files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.write")


class WriteTool(BaseTool):
    """Create or overwrite files, creating parent directories as needed."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, "
            "or overwrites it if it does. Parent directories are created automatically."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return "Error: file_path is required."

        resolved = self._resolve_path(file_path)

        try:
            # Create parent directories if they don't exist
            resolved.parent.mkdir(parents=True, exist_ok=True)

            existed = resolved.exists()
            old_size = resolved.stat().st_size if existed else 0

            resolved.write_text(content, encoding="utf-8")

            new_size = resolved.stat().st_size
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            if existed:
                logger.debug("Overwrote %s (%d -> %d bytes)", resolved, old_size, new_size)
                return (
                    f"Wrote {resolved} "
                    f"({line_count} lines, {new_size} bytes, overwrote {old_size} bytes)"
                )
            else:
                logger.debug("Created %s (%d bytes)", resolved, new_size)
                return f"Created {resolved} ({line_count} lines, {new_size} bytes)"

        except PermissionError:
            return f"Error: permission denied writing to {resolved}"
        except OSError as e:
            logger.warning("Failed to write %s: %s", resolved, e)
            return f"Error writing file: {e}"

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p
