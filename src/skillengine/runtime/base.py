"""
Base skill runtime interface.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of a skill execution."""

    success: bool
    output: str = ""
    error: str | None = None
    exit_code: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, output: str, duration_ms: float = 0.0) -> ExecutionResult:
        """Create a successful result."""
        return cls(success=True, output=output, duration_ms=duration_ms)

    @classmethod
    def error_result(
        cls,
        error: str,
        exit_code: int = 1,
        output: str = "",
        duration_ms: float = 0.0,
    ) -> ExecutionResult:
        """Create an error result."""
        return cls(
            success=False,
            output=output,
            error=error,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )


# Callback type for streaming tool output
OutputCallback = Callable[[str], None]


class SkillRuntime(ABC):
    """
    Abstract base class for skill execution runtimes.

    A runtime is responsible for executing skill commands in a specific
    environment (bash, Python, sandbox, etc.).
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute a command.

        Args:
            command: Command to execute
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout in seconds
            on_output: Callback invoked with each line of output as it arrives.
                       Enables streaming feedback during long-running commands.
            abort_signal: When set, the runtime should terminate the command
                          as soon as possible.

        Returns:
            ExecutionResult with output or error
        """
        pass

    @abstractmethod
    async def execute_script(
        self,
        script: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute a multi-line script.

        Args:
            script: Script content
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout in seconds
            on_output: Callback invoked with each line of output as it arrives.
            abort_signal: When set, the runtime should terminate the script
                          as soon as possible.

        Returns:
            ExecutionResult with output or error
        """
        pass

    def _timer(self) -> _Timer:
        """Create a timer for measuring execution duration."""
        return _Timer()


class _Timer:
    """Simple timer for measuring execution duration."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000
