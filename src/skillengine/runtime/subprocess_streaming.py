"""Shared subprocess streaming collector with timeout and abort support."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from skillengine.runtime.base import ExecutionResult, OutputCallback


class TimerLike(Protocol):
    """Timer protocol used by runtimes."""

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds."""
        ...


async def collect_subprocess_streaming(
    *,
    process: asyncio.subprocess.Process,
    timer: TimerLike,
    timeout: float,
    on_output: OutputCallback | None,
    abort_signal: asyncio.Event | None,
    label: str,
    truncate: Callable[[str], str],
) -> ExecutionResult:
    """Collect subprocess output while supporting timeout and cooperative abort."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    aborted = False
    abort_task: asyncio.Task[None] | None = None

    async def _read_stream(
        stream: asyncio.StreamReader | None,
        lines: list[str],
        callback: OutputCallback | None,
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace")
            lines.append(decoded)
            if callback:
                callback(decoded)

    async def _watch_abort() -> None:
        nonlocal aborted
        if abort_signal is None:
            return
        await abort_signal.wait()
        aborted = True
        try:
            process.kill()
        except ProcessLookupError:
            pass

    try:
        if abort_signal is not None and abort_signal.is_set():
            aborted = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return ExecutionResult.error_result(
                error="Aborted",
                exit_code=-2,
                output=truncate("".join(stdout_lines)),
                duration_ms=timer.elapsed_ms(),
            )

        # Wait on stdout/stderr readers only; abort watcher is independent.
        reader_tasks = [
            asyncio.create_task(_read_stream(process.stdout, stdout_lines, on_output)),
            asyncio.create_task(_read_stream(process.stderr, stderr_lines, None)),
        ]
        if abort_signal is not None:
            abort_task = asyncio.create_task(_watch_abort())

        _, pending = await asyncio.wait(reader_tasks, timeout=timeout)
        readers_done = len(pending) == 0

        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

        if not readers_done and not aborted:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return ExecutionResult.error_result(
                error=f"{label} timed out after {timeout}s",
                exit_code=-1,
                output=truncate("".join(stdout_lines)),
                duration_ms=timer.elapsed_ms(),
            )

        await process.wait()

        if aborted:
            return ExecutionResult.error_result(
                error="Aborted",
                exit_code=-2,
                output=truncate("".join(stdout_lines)),
                duration_ms=timer.elapsed_ms(),
            )

        output = truncate("".join(stdout_lines))
        error_output = "".join(stderr_lines)
        if process.returncode == 0:
            return ExecutionResult.success_result(
                output=output,
                duration_ms=timer.elapsed_ms(),
            )

        return ExecutionResult.error_result(
            error=error_output or f"{label} failed with exit code {process.returncode}",
            exit_code=process.returncode or 1,
            output=output,
            duration_ms=timer.elapsed_ms(),
        )
    except Exception as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return ExecutionResult.error_result(
            error=str(exc),
            exit_code=-1,
            duration_ms=timer.elapsed_ms(),
        )
    finally:
        if abort_task is not None:
            if not abort_task.done():
                abort_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await abort_task
