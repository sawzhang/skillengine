"""Edit tool - exact string replacement in files."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.edit")


class EditTool(BaseTool):
    """Perform exact string replacement in files."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return (
            "Perform exact string replacements in a file. The old_string must appear "
            "in the file and will be replaced with new_string. By default, old_string "
            "must be unique in the file. Set replace_all=true to replace every occurrence."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "If true, replace all occurrences. If false (default), "
                        "old_string must appear exactly once."
                    ),
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = args.get("replace_all", False)

        if not file_path:
            return "Error: file_path is required."
        if not old_string:
            return "Error: old_string is required and cannot be empty."
        if old_string == new_string:
            return "Error: old_string and new_string must be different."

        resolved = self._resolve_path(file_path)

        if not resolved.exists():
            return f"Error: file not found: {resolved}"

        if resolved.is_dir():
            return f"Error: {resolved} is a directory, not a file."

        try:
            original = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        # Count occurrences
        count = original.count(old_string)

        if count == 0:
            # Provide a helpful hint with close matches
            hint = self._find_close_match(original, old_string)
            msg = f"Error: old_string not found in {resolved.name}."
            if hint:
                msg += f"\n\nDid you mean:\n{hint}"
            return msg

        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in {resolved.name}. "
                f"Provide more surrounding context to make it unique, "
                f"or set replace_all=true to replace all occurrences."
            )

        # Perform the replacement
        if replace_all:
            updated = original.replace(old_string, new_string)
        else:
            # Replace only the first (and only) occurrence
            updated = original.replace(old_string, new_string, 1)

        try:
            resolved.write_text(updated, encoding="utf-8")
        except Exception as e:
            return f"Error writing file: {e}"

        # Build a unified diff preview
        diff = self._make_diff(original, updated, resolved.name)

        replaced_word = "replacements" if count > 1 else "replacement"
        logger.debug(
            "Edited %s: %d %s",
            resolved,
            count,
            replaced_word,
        )
        return f"Edited {resolved} ({count} {replaced_word})\n\n{diff}"

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path, making it absolute if needed."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(self.cwd) / p

    def _make_diff(self, original: str, updated: str, filename: str) -> str:
        """Generate a unified diff between original and updated content."""
        original_lines = original.splitlines(keepends=True)
        updated_lines = updated.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )

        if not diff_lines:
            return "(no visible diff)"

        # Limit diff output to avoid overwhelming the LLM
        max_diff_lines = 100
        if len(diff_lines) > max_diff_lines:
            diff_lines = diff_lines[:max_diff_lines]
            diff_lines.append(f"\n... ({len(diff_lines)} more lines)")

        return "".join(line.rstrip("\n") + "\n" for line in diff_lines)

    def _find_close_match(self, content: str, target: str) -> str | None:
        """Try to find a close match to the target string in the file content."""
        # Only attempt for reasonably-sized targets to avoid performance issues
        if len(target) > 500 or len(target) < 3:
            return None

        # Search line by line for lines that partially match
        target_lines = target.splitlines()
        if not target_lines:
            return None

        first_target_line = target_lines[0].strip()
        if len(first_target_line) < 3:
            return None

        content_lines = content.splitlines()
        matches = difflib.get_close_matches(
            first_target_line,
            [line.strip() for line in content_lines],
            n=1,
            cutoff=0.6,
        )

        if matches:
            return matches[0]
        return None
