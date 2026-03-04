"""Apply-patch tool based on OpenAI-style operation payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillkit.logging import get_logger
from skillkit.tools.apply_diff import apply_diff
from skillkit.tools.registry import BaseTool

logger = get_logger("tools.apply_patch")

@dataclass
class _ApplyPatchOperation:
    kind: str  # create_file | update_file | delete_file
    path: str
    diff: str | None = None


class ApplyPatchTool(BaseTool):
    """Apply a single file operation with OpenAI apply_patch payload shape."""

    def __init__(self, cwd: str | None = None, *, enforce_workspace_boundary: bool = True) -> None:
        super().__init__(cwd)
        self.enforce_workspace_boundary = enforce_workspace_boundary

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Apply a single file patch operation. Provide type as "
            "create_file/update_file/delete_file and path as the target file path. "
            "For create_file and update_file, diff is required and must use "
            "line prefixes (+ to add, - to remove, space for context in updates). "
            "For create_file, all diff lines must start with +."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["create_file", "update_file", "delete_file"],
                    "description": "Type of mutation to apply.",
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Target file path.",
                },
                "diff": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Unified-like diff body for create/update operations, "
                        "for example '-old\\n+new'."
                    ),
                },
            },
            "required": ["type", "path"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "Error: apply_patch arguments must be a JSON object."

        try:
            operation = self._coerce_operation(args)
            result = self._apply_operation(operation)
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # pragma: no cover - safety net
            return f"Error applying patch: {exc}"

        logger.debug("Applied operation %s on %s", operation.kind, operation.path)
        return result

    def _coerce_operation(self, payload: dict[str, Any]) -> _ApplyPatchOperation:
        op_type_value = str(payload.get("type") or "")
        if op_type_value not in {"create_file", "update_file", "delete_file"}:
            raise ValueError(f"unknown apply_patch operation: {op_type_value or '<empty>'}")

        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("apply_patch operation requires a valid path")

        diff_value = payload.get("diff")
        if op_type_value in {"create_file", "update_file"}:
            if not isinstance(diff_value, str) or not diff_value:
                raise ValueError(f"apply_patch operation {op_type_value} requires non-empty diff")
            diff: str | None = diff_value
        else:
            diff = None

        return _ApplyPatchOperation(kind=op_type_value, path=path.strip(), diff=diff)

    def _apply_operation(self, operation: _ApplyPatchOperation) -> str:
        source_path = self._resolve_path(operation.path)

        if operation.kind == "create_file":
            if source_path.exists():
                raise ValueError(f"cannot create {operation.path}: file already exists")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            content = apply_diff("", operation.diff or "", mode="create")
            source_path.write_text(content, encoding="utf-8")
            return f"Created {source_path}"

        if operation.kind == "delete_file":
            if not source_path.exists():
                raise ValueError(f"cannot delete {operation.path}: file not found")
            if source_path.is_dir():
                raise ValueError(f"cannot delete {operation.path}: path is a directory")
            source_path.unlink()
            return f"Deleted {source_path}"

        if operation.kind == "update_file":
            if not source_path.exists():
                raise ValueError(f"cannot update {operation.path}: file not found")
            if source_path.is_dir():
                raise ValueError(f"cannot update {operation.path}: path is a directory")

            original = source_path.read_text(encoding="utf-8", errors="replace")
            updated = apply_diff(original, operation.diff or "", mode="default")
            source_path.write_text(updated, encoding="utf-8")
            return f"Updated {source_path}"

        raise ValueError(f"unsupported operation kind: {operation.kind}")

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve patch path and optionally enforce workspace boundaries."""
        if not raw_path:
            raise ValueError("empty patch file path")

        workspace = Path(self.cwd).resolve()
        candidate = Path(raw_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (workspace / candidate).resolve()
        )

        if self.enforce_workspace_boundary:
            try:
                resolved.relative_to(workspace)
            except ValueError as exc:
                raise ValueError(f"path escapes workspace: {raw_path}") from exc
        return resolved
