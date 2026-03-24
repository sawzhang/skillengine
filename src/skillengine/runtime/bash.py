"""
Bash execution runtime with streaming output and abort support.
"""

from __future__ import annotations

import asyncio
import os

from skillengine.runtime.base import ExecutionResult, OutputCallback, SkillRuntime
from skillengine.runtime.subprocess_streaming import collect_subprocess_streaming


class BashRuntime(SkillRuntime):
    """
    Bash-based skill execution runtime.

    Executes commands using the system shell. Supports streaming output
    via ``on_output`` callback and cooperative cancellation via ``abort_signal``.
    """

    def __init__(
        self,
        shell: str = "/bin/bash",
        default_timeout: float = 30.0,
        max_output_size: int = 1_000_000,  # 1MB
    ) -> None:
        self.shell = shell
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Execute a single command with optional streaming and abort."""
        timer = self._timer()
        timeout = timeout or self.default_timeout
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )

            return await self._collect_output(
                process, timer, timeout, on_output, abort_signal, label="Command"
            )

        except Exception as e:
            return ExecutionResult.error_result(
                error=str(e),
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

    async def execute_script(
        self,
        script: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Execute a multi-line script with optional streaming and abort."""
        timer = self._timer()
        timeout = timeout or self.default_timeout
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            process = await asyncio.create_subprocess_exec(
                self.shell,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )

            return await self._collect_output(
                process, timer, timeout, on_output, abort_signal, label="Script"
            )

        except Exception as e:
            return ExecutionResult.error_result(
                error=str(e),
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

    async def _collect_output(
        self,
        process: asyncio.subprocess.Process,
        timer: object,
        timeout: float,
        on_output: OutputCallback | None,
        abort_signal: asyncio.Event | None,
        label: str = "Command",
    ) -> ExecutionResult:
        """
        Collect output from a subprocess.

        When ``on_output`` is provided, reads stdout line by line and invokes
        the callback with each line for real-time streaming. Otherwise falls
        back to the efficient ``communicate()`` approach.

        When ``abort_signal`` is set, kills the process immediately.
        """
        if on_output is None and abort_signal is None:
            # Fast path: no streaming, no abort — use communicate()
            return await self._collect_simple(process, timer, timeout, label)

        return await collect_subprocess_streaming(
            process=process,
            timer=timer,
            timeout=timeout,
            on_output=on_output,
            abort_signal=abort_signal,
            label=label,
            truncate=self._truncate,
        )

    async def _collect_simple(
        self,
        process: asyncio.subprocess.Process,
        timer: object,
        timeout: float,
        label: str,
    ) -> ExecutionResult:
        """Fast path: collect output using communicate()."""
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecutionResult.error_result(
                error=f"{label} timed out after {timeout}s",
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

        output = self._decode_output(stdout)
        error_output = self._decode_output(stderr)

        if process.returncode == 0:
            return ExecutionResult.success_result(
                output=output,
                duration_ms=timer.elapsed_ms(),
            )
        else:
            return ExecutionResult.error_result(
                error=error_output or f"{label} failed with exit code {process.returncode}",
                exit_code=process.returncode or 1,
                output=output,
                duration_ms=timer.elapsed_ms(),
            )

    def _decode_output(self, data: bytes) -> str:
        """Decode command output, truncating if necessary."""
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        return self._truncate(text)

    def _truncate(self, text: str) -> str:
        """Truncate text if it exceeds max_output_size."""
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + "\n... (output truncated)"
        return text
