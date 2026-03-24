"""Bash tool - execute shell commands."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from skillengine.logging import get_logger
from skillengine.tools.registry import BaseTool

logger = get_logger("tools.bash")

# Default timeout in seconds
_DEFAULT_TIMEOUT = 120.0

# Maximum output size in characters before truncation
_MAX_OUTPUT = 100_000


class BashTool(BaseTool):
    """Execute shell commands and return their output."""

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output (stdout and stderr combined). "
            "Commands run in the working directory. Use this for git, npm, docker, "
            "and other terminal operations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (f"Timeout in seconds. Defaults to {int(_DEFAULT_TIMEOUT)}."),
                },
            },
            "required": ["command"],
        }

    async def execute(self, args: dict[str, Any]) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", _DEFAULT_TIMEOUT)

        if not command:
            return "Error: command is required."

        if not isinstance(timeout, (int, float)):
            timeout = _DEFAULT_TIMEOUT
        timeout = max(1, min(timeout, 600))  # Clamp between 1s and 10min

        logger.debug("Executing: %s (cwd=%s, timeout=%ss)", command, self.cwd, timeout)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=os.environ.copy(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                return f"Error: command timed out after {timeout}s"

        except FileNotFoundError:
            return f"Error: command not found or cwd does not exist: {self.cwd}"
        except PermissionError:
            return "Error: permission denied"
        except Exception as e:
            logger.warning("Command execution failed: %s", e)
            return f"Error: {e}"

        stdout_str = self._decode(stdout)
        stderr_str = self._decode(stderr)

        # Combine stdout and stderr
        parts: list[str] = []
        if stdout_str:
            parts.append(stdout_str)
        if stderr_str:
            # Only label stderr if there's also stdout
            if stdout_str:
                parts.append(f"STDERR:\n{stderr_str}")
            else:
                parts.append(stderr_str)

        output = "\n".join(parts) if parts else "(no output)"

        # Truncate if too long
        output = self._truncate(output)

        exit_code = process.returncode or 0
        if exit_code != 0:
            output = f"Exit code: {exit_code}\n{output}"

        logger.debug("Command finished (exit=%d, %d chars)", exit_code, len(output))
        return output

    @staticmethod
    def _decode(data: bytes) -> str:
        """Decode subprocess output bytes to string."""
        try:
            return data.decode("utf-8", errors="replace").rstrip()
        except Exception:
            return str(data)

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate output if it exceeds the maximum size."""
        if len(text) > _MAX_OUTPUT:
            half = _MAX_OUTPUT // 2
            return (
                text[:half]
                + f"\n\n... ({len(text) - _MAX_OUTPUT} characters truncated) ...\n\n"
                + text[-half:]
            )
        return text
