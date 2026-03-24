"""BoxLite sandbox runtime with VM-level execution isolation."""

from __future__ import annotations

import asyncio
import uuid
from enum import Enum

from skillengine.runtime.base import ExecutionResult, OutputCallback, SkillRuntime

try:
    import boxlite
except ImportError as e:
    raise ImportError(
        "BoxLite SDK is required for BoxLiteRuntime. "
        "Install it with: pip install 'skillengine[sandbox]'"
    ) from e


class SecurityLevel(Enum):
    """Security level presets for BoxLite sandbox resources."""

    DEV = "dev"
    STANDARD = "standard"
    MAXIMUM = "maximum"


class BoxLiteRuntime(SkillRuntime):
    """
    BoxLite-based skill execution runtime with VM-level isolation.

    Uses BoxLite micro-VMs to execute commands in a hardware-isolated sandbox.
    The box is lazily initialized on first use and persists across executions
    (packages, files, and environment are retained).

    Example::

        async with BoxLiteRuntime() as runtime:
            result = await runtime.execute("echo hello")

        # Or with explicit lifecycle:
        runtime = BoxLiteRuntime()
        await runtime.start()
        result = await runtime.execute("pip install requests && python -c 'import requests'")
        await runtime.destroy()
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        image: str = "python:slim",
        memory_mib: int = 512,
        cpus: int = 1,
        default_timeout: float = 30.0,
        max_output_size: int = 1_000_000,
        auto_destroy: bool = True,
        volumes: list[tuple[str, str, str]] | None = None,
        working_dir: str | None = None,
        box_env: list[tuple[str, str]] | None = None,
    ) -> None:
        self.security_level = security_level
        self.image = image
        self.memory_mib = memory_mib
        self.cpus = cpus
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.auto_destroy = auto_destroy
        self.volumes = volumes
        self.working_dir = working_dir
        self.box_env = box_env

        self._boxlite_runtime: boxlite.Boxlite | None = None
        self._box: boxlite.Box | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lazy box initialization
    # ------------------------------------------------------------------

    async def _ensure_box(self) -> boxlite.Box:
        """Lazily initialize the BoxLite sandbox (double-checked locking)."""
        if self._box is not None:
            return self._box

        async with self._lock:
            if self._box is not None:
                return self._box

            self._boxlite_runtime = boxlite.Boxlite.default()
            box_options = self._resolve_box_options()
            self._box = await self._boxlite_runtime.create(box_options)
            return self._box

    def _resolve_box_options(self) -> boxlite.BoxOptions:
        """Map security level to BoxOptions."""
        # Common kwargs shared across all security levels
        common: dict[str, object] = {"image": self.image, "auto_remove": False}
        if self.volumes is not None:
            common["volumes"] = self.volumes
        if self.working_dir is not None:
            common["working_dir"] = self.working_dir
        if self.box_env is not None:
            common["env"] = self.box_env

        if self.security_level == SecurityLevel.DEV:
            return boxlite.BoxOptions(
                cpus=max(self.cpus, 2),
                memory_mib=max(self.memory_mib, 1024),
                **common,
            )
        elif self.security_level == SecurityLevel.MAXIMUM:
            return boxlite.BoxOptions(
                cpus=1,
                memory_mib=256,
                **common,
            )
        else:  # STANDARD
            return boxlite.BoxOptions(
                cpus=self.cpus,
                memory_mib=self.memory_mib,
                **common,
            )

    # ------------------------------------------------------------------
    # SkillRuntime interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Execute a command in the BoxLite sandbox."""
        timer = self._timer()
        timeout = timeout or self.default_timeout

        try:
            box = await self._ensure_box()
            full_command = self._build_command(command, cwd, env)

            return await self._run_with_timeout_and_abort(
                box, full_command, timer, timeout, on_output, abort_signal
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
        """Execute a multi-line script in the BoxLite sandbox."""
        timer = self._timer()
        timeout = timeout or self.default_timeout

        try:
            box = await self._ensure_box()

            # Write script to temp file in the box, then execute it
            script_path = f"/tmp/_skillengine_{uuid.uuid4().hex[:8]}.sh"
            write_cmd = (
                f"cat > {script_path} << 'SKILLKIT_SCRIPT_EOF'\n{script}\nSKILLKIT_SCRIPT_EOF"
            )
            write_exec = await box.exec("bash", "-c", write_cmd)
            await write_exec.wait()

            chmod_exec = await box.exec("chmod", "+x", script_path)
            await chmod_exec.wait()

            # Build run command with optional cwd/env
            run_cmd = self._build_command(f"bash {script_path}", cwd, env)

            result = await self._run_with_timeout_and_abort(
                box, run_cmd, timer, timeout, on_output, abort_signal
            )

            # Cleanup temp file (best-effort)
            try:
                cleanup = await box.exec("rm", "-f", script_path)
                await cleanup.wait()
            except Exception:
                pass

            return result
        except Exception as e:
            return ExecutionResult.error_result(
                error=str(e),
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Build a shell command string with optional cwd and env prefix."""
        parts: list[str] = []
        if env:
            for key, value in env.items():
                escaped_value = value.replace("'", "'\\''")
                parts.append(f"export {key}='{escaped_value}'")
        if cwd:
            parts.append(f"cd {cwd}")
        parts.append(command)
        return " && ".join(parts)

    async def _run_with_timeout_and_abort(
        self,
        box: boxlite.Box,
        command: str,
        timer: object,
        timeout: float,
        on_output: OutputCallback | None,
        abort_signal: asyncio.Event | None,
    ) -> ExecutionResult:
        """Run a command with unified timeout and abort handling."""
        execution = await box.exec("bash", "-c", command)

        async def _do_execution() -> boxlite.ExecResult:
            result = await execution.wait()
            # Post-completion streaming: deliver lines via on_output callback
            # TODO: use execution.stdout() async iterator for true real-time streaming
            if on_output is not None and result.stdout:
                for line in result.stdout.splitlines(keepends=True):
                    on_output(line)
            return result

        exec_task = asyncio.create_task(_do_execution())

        try:
            # Handle pre-set abort
            if abort_signal is not None and abort_signal.is_set():
                exec_task.cancel()
                try:
                    await execution.kill()
                except Exception:
                    pass
                return ExecutionResult.error_result(
                    error="Aborted",
                    exit_code=-2,
                    duration_ms=timer.elapsed_ms(),
                )

            if abort_signal is not None:
                # Race between: execution completion, abort signal, timeout
                abort_task = asyncio.create_task(abort_signal.wait())
                timeout_task = asyncio.create_task(asyncio.sleep(timeout))

                done, pending = await asyncio.wait(
                    [exec_task, abort_task, timeout_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except (asyncio.CancelledError, Exception):
                        pass

                if exec_task in done:
                    return self._map_result(exec_task.result(), timer)
                elif abort_task in done:
                    try:
                        await execution.kill()
                    except Exception:
                        pass
                    return ExecutionResult.error_result(
                        error="Aborted",
                        exit_code=-2,
                        duration_ms=timer.elapsed_ms(),
                    )
                else:
                    # Timeout
                    try:
                        await execution.kill()
                    except Exception:
                        pass
                    return ExecutionResult.error_result(
                        error=f"Command timed out after {timeout}s",
                        exit_code=-1,
                        duration_ms=timer.elapsed_ms(),
                    )
            else:
                # No abort signal — just apply timeout
                box_result = await asyncio.wait_for(exec_task, timeout=timeout)
                return self._map_result(box_result, timer)

        except asyncio.TimeoutError:
            exec_task.cancel()
            try:
                await execution.kill()
            except Exception:
                pass
            return ExecutionResult.error_result(
                error=f"Command timed out after {timeout}s",
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )
        except asyncio.CancelledError:
            try:
                await execution.kill()
            except Exception:
                pass
            raise

    def _map_result(
        self,
        box_result: boxlite.ExecResult,
        timer: object,
    ) -> ExecutionResult:
        """Map BoxLite ExecResult to SkillEngine ExecutionResult."""
        stdout = self._truncate(box_result.stdout or "")
        stderr = box_result.stderr or ""

        if box_result.exit_code == 0:
            return ExecutionResult.success_result(
                output=stdout,
                duration_ms=timer.elapsed_ms(),
            )
        else:
            return ExecutionResult.error_result(
                error=stderr or f"Command failed with exit code {box_result.exit_code}",
                exit_code=box_result.exit_code,
                output=stdout,
                duration_ms=timer.elapsed_ms(),
            )

    def _truncate(self, text: str) -> str:
        """Truncate text if it exceeds max_output_size."""
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + "\n... (output truncated)"
        return text

    # ------------------------------------------------------------------
    # File I/O (BoxLite-specific, not part of SkillRuntime ABC)
    # ------------------------------------------------------------------

    async def read_file(self, path: str) -> str:
        """Read a file from inside the sandbox."""
        box = await self._ensure_box()
        execution = await box.exec("cat", path)
        result = await execution.wait()
        if result.exit_code != 0:
            raise FileNotFoundError(
                result.stderr or f"Failed to read {path} (exit {result.exit_code})"
            )
        return result.stdout or ""

    async def write_file(self, path: str, content: str) -> None:
        """Write a file inside the sandbox."""
        box = await self._ensure_box()
        escaped = content.replace("'", "'\\''")
        cmd = f"mkdir -p $(dirname {path}) && printf '%s' '{escaped}' > {path}"
        execution = await box.exec("bash", "-c", cmd)
        result = await execution.wait()
        if result.exit_code != 0:
            raise OSError(result.stderr or f"Failed to write {path} (exit {result.exit_code})")

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists inside the sandbox."""
        box = await self._ensure_box()
        execution = await box.exec("test", "-e", path)
        result = await execution.wait()
        return result.exit_code == 0

    async def list_dir(self, path: str) -> str:
        """List directory contents inside the sandbox."""
        box = await self._ensure_box()
        execution = await box.exec("ls", "-la", path)
        result = await execution.wait()
        if result.exit_code != 0:
            raise FileNotFoundError(
                result.stderr or f"Failed to list {path} (exit {result.exit_code})"
            )
        return result.stdout or ""

    async def is_ready(self) -> bool:
        """Check if the sandbox is ready to accept commands."""
        if self._box is None:
            return False
        try:
            execution = await self._box.exec("echo", "ready")
            result = await execution.wait()
            return result.exit_code == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Eagerly initialize the sandbox."""
        await self._ensure_box()

    async def stop(self) -> None:
        """Stop the sandbox (box can be restarted later)."""
        if self._box is not None:
            try:
                await self._box.stop()
            except Exception:
                pass

    async def destroy(self) -> None:
        """Destroy the sandbox and release all resources."""
        if self._box is not None:
            try:
                await self._box.remove()
            except Exception:
                pass
            self._box = None

    async def __aenter__(self) -> BoxLiteRuntime:
        await self._ensure_box()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self.auto_destroy:
            await self.destroy()
