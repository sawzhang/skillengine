"""
Code Mode runtime: search + execute pattern for LLM agents.

Inspired by Cloudflare's code-mode-mcp design. Instead of exposing many
individual tools/commands, this runtime accepts Python code from the LLM
and executes it with injected context objects:

- ``spec``: Read-only data (API schemas, OpenAPI specs, configs) for discovery
- ``ctx``: Client objects and utilities for execution

The LLM writes Python code like::

    # Search phase — discover endpoints
    [p for p in spec['paths'] if '/users' in p]

    # Execute phase — call the API
    result = ctx['client'].get('/users/123')

This reduces token usage from O(N) (one tool per endpoint) to O(1)
(two tools regardless of API surface area).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import textwrap
from typing import Any

from skillkit.runtime.base import ExecutionResult, OutputCallback, SkillRuntime

# Sentinel for "result not set by user code"
_UNSET = object()

# Safe builtins exposed to code-mode execution
_SAFE_BUILTINS: dict[str, Any] = {
    # Types
    "bool": bool,
    "bytes": bytes,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "set": set,
    "str": str,
    "tuple": tuple,
    # Iteration
    "enumerate": enumerate,
    "filter": filter,
    "map": map,
    "range": range,
    "reversed": reversed,
    "sorted": sorted,
    "zip": zip,
    # Inspection
    "callable": callable,
    "dir": dir,
    "getattr": getattr,
    "hasattr": hasattr,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "type": type,
    "vars": vars,
    # Math
    "abs": abs,
    "divmod": divmod,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sum": sum,
    # Logic
    "all": all,
    "any": any,
    # Representation
    "repr": repr,
    "format": format,
    "print": print,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Exceptions (needed for try/except in user code)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ImportError": ImportError,
}


class CodeModeRuntime(SkillRuntime):
    """
    Code Mode runtime implementing the search + execute pattern.

    Instead of exposing many individual commands/tools, this runtime lets
    the LLM write Python code that runs against injected data and clients.

    Two execution modes:

    - **search**: Code runs with ``spec`` in scope. For API discovery,
      schema exploration, and filtering. Read-only by convention.
    - **execute**: Code runs with both ``spec`` and ``ctx`` in scope.
      For calling APIs, mutating state, and performing actions.

    Example::

        import httpx

        runtime = CodeModeRuntime(
            spec=openapi_spec,           # dict loaded from OpenAPI JSON
            ctx={"http": httpx.Client()}, # injected client
        )

        # LLM discovers endpoints
        result = await runtime.search(
            "[p for p in spec['paths'] if '/users' in p]"
        )

        # LLM calls the API
        result = await runtime.run(
            "result = ctx['http'].get(spec['servers'][0]['url'] + '/users')"
        )

    Args:
        spec: Data structure for search/discovery (any Python object).
        ctx: Dict of objects injected into execute mode. Keys become
            variable names accessible in user code via ``ctx['key']``.
        allowed_modules: Module names the code is allowed to import.
            Defaults to ``["json", "re", "math", "datetime", "collections",
            "itertools", "functools", "urllib.parse"]``.
        sandbox: Execution mode. ``"inprocess"`` uses ``exec()`` with
            restricted builtins (flexible, works with any objects).
            ``"subprocess"`` runs code in a child process (more isolated,
            only works with JSON-serializable spec/ctx).
        default_timeout: Default execution timeout in seconds.
        max_output_size: Maximum output size in bytes before truncation.
    """

    DEFAULT_ALLOWED_MODULES = [
        "json",
        "re",
        "math",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "urllib.parse",
    ]

    def __init__(
        self,
        spec: Any = None,
        ctx: dict[str, Any] | None = None,
        allowed_modules: list[str] | None = None,
        sandbox: str = "inprocess",  # "inprocess" or "subprocess"
        default_timeout: float = 30.0,
        max_output_size: int = 1_000_000,
    ) -> None:
        self.spec = spec
        self.ctx = ctx or {}
        self.allowed_modules = allowed_modules or self.DEFAULT_ALLOWED_MODULES
        self.sandbox = sandbox
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size

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
        """Execute Python code with full context (spec + ctx)."""
        return await self._run_code(
            code=command,
            mode="execute",
            cwd=cwd,
            env=env,
            timeout=timeout,
            on_output=on_output,
            abort_signal=abort_signal,
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
        """Execute multi-line Python code with full context (spec + ctx)."""
        return await self._run_code(
            code=script,
            mode="execute",
            cwd=cwd,
            env=env,
            timeout=timeout,
            on_output=on_output,
            abort_signal=abort_signal,
        )

    # ------------------------------------------------------------------
    # Code Mode specific methods
    # ------------------------------------------------------------------

    async def search(
        self,
        code: str,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute code in search mode (spec only, no ctx).

        Use this for API discovery, schema exploration, and filtering.
        The ``spec`` variable is available in the code's namespace.

        Args:
            code: Python code to execute. Assign to ``result`` to return
                structured data, or use ``print()`` for text output.
            timeout: Execution timeout in seconds.
            on_output: Callback for streaming output lines.
            abort_signal: Event to signal abort.

        Returns:
            ExecutionResult with the code's output.
        """
        return await self._run_code(
            code=code,
            mode="search",
            timeout=timeout,
            on_output=on_output,
            abort_signal=abort_signal,
        )

    async def run(
        self,
        code: str,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute code in execute mode (spec + ctx).

        Use this for calling APIs, mutating state, and performing actions.
        Both ``spec`` and ``ctx`` are available in the code's namespace.

        Args:
            code: Python code to execute. Assign to ``result`` to return
                structured data, or use ``print()`` for text output.
            timeout: Execution timeout in seconds.
            on_output: Callback for streaming output lines.
            abort_signal: Event to signal abort.

        Returns:
            ExecutionResult with the code's output.
        """
        return await self._run_code(
            code=code,
            mode="execute",
            timeout=timeout,
            on_output=on_output,
            abort_signal=abort_signal,
        )

    # ------------------------------------------------------------------
    # Tool definitions for LLM adapters
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Generate ``search`` and ``execute`` tool definitions for LLM.

        Returns tool definitions in OpenAI function-calling format.
        These can be passed to an LLM adapter so the model can call
        ``search`` and ``execute`` as tools.

        Returns:
            List of tool definition dicts.
        """
        tools: list[dict[str, Any]] = []

        # Build spec hint
        spec_hint = ""
        if self.spec is not None:
            if isinstance(self.spec, dict):
                top_keys = list(self.spec.keys())[:10]
                spec_hint = f" Top-level keys: {top_keys}."
            elif isinstance(self.spec, list):
                spec_hint = f" The spec is a list with {len(self.spec)} items."

        if self.spec is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": (
                            "Search the API spec by writing Python code. "
                            "The `spec` variable contains the full specification."
                            f"{spec_hint} "
                            "Assign to `result` to return structured data, "
                            "or use `print()` for text output."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": (
                                        "Python code that accesses the `spec` variable. "
                                        "Example: `result = [k for k in spec['paths'] "
                                        "if '/users' in k]`"
                                    ),
                                }
                            },
                            "required": ["code"],
                        },
                    },
                }
            )

        ctx_names = list(self.ctx.keys()) if self.ctx else []
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": (
                        "Execute Python code with injected context objects. "
                        f"Available in `ctx`: {ctx_names}. "
                        "The `spec` variable is also available. "
                        "Assign to `result` to return structured data, "
                        "or use `print()` for text output."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code that uses `ctx` and optionally `spec`. "
                                    "Example: `result = ctx['client'].get('/users')`"
                                ),
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        )

        return tools

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run_code(
        self,
        code: str,
        mode: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: OutputCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Route to the appropriate execution backend."""
        timeout = timeout or self.default_timeout

        # Check abort before starting
        if abort_signal is not None and abort_signal.is_set():
            return ExecutionResult.error_result(error="Aborted", exit_code=-2)

        if self.sandbox == "subprocess":
            return await self._run_subprocess(
                code, mode, cwd, env, timeout, on_output, abort_signal
            )
        else:
            return await self._run_inprocess(
                code, mode, timeout, on_output, abort_signal
            )

    async def _run_inprocess(
        self,
        code: str,
        mode: str,
        timeout: float,
        on_output: OutputCallback | None,
        abort_signal: asyncio.Event | None,
    ) -> ExecutionResult:
        """Execute code in-process using exec() with restricted namespace."""
        timer = self._timer()

        namespace = self._build_namespace(mode)
        stdout_buf = io.StringIO()

        def _exec() -> str:
            with contextlib.redirect_stdout(stdout_buf):
                compiled = compile(code, "<code-mode>", "exec")
                exec(compiled, namespace)  # noqa: S102

            # Prefer explicit result variable over stdout
            if "result" in namespace and namespace["result"] is not _UNSET:
                val = namespace["result"]
                if isinstance(val, str):
                    return val
                try:
                    return json.dumps(val, default=str, ensure_ascii=False, indent=2)
                except (TypeError, ValueError):
                    return str(val)
            return stdout_buf.getvalue()

        try:
            # Run in executor to support timeout and avoid blocking
            loop = asyncio.get_running_loop()

            if abort_signal is not None:
                # Run with abort watching
                exec_future = loop.run_in_executor(None, _exec)
                abort_future = asyncio.ensure_future(abort_signal.wait())

                done, pending = await asyncio.wait(
                    [exec_future, abort_future],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except (asyncio.CancelledError, Exception):
                        pass

                if exec_future in done:
                    output = exec_future.result()
                elif abort_signal.is_set():
                    return ExecutionResult.error_result(
                        error="Aborted",
                        exit_code=-2,
                        output=self._truncate(stdout_buf.getvalue()),
                        duration_ms=timer.elapsed_ms(),
                    )
                else:
                    return ExecutionResult.error_result(
                        error=f"Code timed out after {timeout}s",
                        exit_code=-1,
                        output=self._truncate(stdout_buf.getvalue()),
                        duration_ms=timer.elapsed_ms(),
                    )
            else:
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, _exec),
                    timeout=timeout,
                )

            output = self._truncate(output)

            if on_output and output:
                for line in output.splitlines(keepends=True):
                    on_output(line)

            return ExecutionResult.success_result(
                output=output,
                duration_ms=timer.elapsed_ms(),
            )

        except asyncio.TimeoutError:
            return ExecutionResult.error_result(
                error=f"Code timed out after {timeout}s",
                exit_code=-1,
                output=self._truncate(stdout_buf.getvalue()),
                duration_ms=timer.elapsed_ms(),
            )
        except Exception as e:
            return ExecutionResult.error_result(
                error=f"{type(e).__name__}: {e}",
                exit_code=1,
                output=self._truncate(stdout_buf.getvalue()),
                duration_ms=timer.elapsed_ms(),
            )

    async def _run_subprocess(
        self,
        code: str,
        mode: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float,
        on_output: OutputCallback | None,
        abort_signal: asyncio.Event | None,
    ) -> ExecutionResult:
        """Execute code in an isolated subprocess."""
        timer = self._timer()

        # Serialize spec and ctx for subprocess (use "null" for None)
        try:
            spec_json = json.dumps(self.spec, default=str)
        except (TypeError, ValueError):
            spec_json = "null"

        try:
            ctx_json = json.dumps(self.ctx, default=str) if self.ctx else "{}"
        except (TypeError, ValueError):
            ctx_json = "{}"

        # Build wrapper script
        # The subprocess itself provides isolation; we just set up the namespace
        wrapper = textwrap.dedent(f"""\
import json, sys

spec = json.loads({json.dumps(spec_json)})
ctx = json.loads({json.dumps(ctx_json)})

_UNSET = object()
result = _UNSET

{"" if mode == "execute" else "ctx = None  # search mode: ctx not available"}

exec(compile({json.dumps(code)}, "<code-mode>", "exec"))

if result is not _UNSET:
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, default=str, ensure_ascii=False, indent=2))
""")

        import os

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                wrapper,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )

            if on_output is None and abort_signal is None:
                return await self._collect_simple(process, timer, timeout)

            return await self._collect_streaming(
                process, timer, timeout, on_output, abort_signal
            )

        except Exception as e:
            return ExecutionResult.error_result(
                error=str(e),
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

    async def _collect_simple(
        self,
        process: asyncio.subprocess.Process,
        timer: object,
        timeout: float,
    ) -> ExecutionResult:
        """Fast path: collect subprocess output using communicate()."""
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecutionResult.error_result(
                error=f"Code timed out after {timeout}s",
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )

        output = self._truncate(stdout.decode("utf-8", errors="replace"))
        error_output = stderr.decode("utf-8", errors="replace")

        if process.returncode == 0:
            return ExecutionResult.success_result(
                output=output,
                duration_ms=timer.elapsed_ms(),
            )
        else:
            return ExecutionResult.error_result(
                error=error_output or f"Code failed with exit code {process.returncode}",
                exit_code=process.returncode or 1,
                output=output,
                duration_ms=timer.elapsed_ms(),
            )

    async def _collect_streaming(
        self,
        process: asyncio.subprocess.Process,
        timer: object,
        timeout: float,
        on_output: OutputCallback | None,
        abort_signal: asyncio.Event | None,
    ) -> ExecutionResult:
        """Streaming path: read output line by line with abort support."""
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
                    duration_ms=timer.elapsed_ms(),
                )

            # Start stdout/stderr readers; abort watcher runs independently so it
            # cannot keep the wait() call pending until timeout.
            reader_tasks = [
                asyncio.create_task(_read_stream(process.stdout, stdout_lines, on_output)),
                asyncio.create_task(_read_stream(process.stderr, stderr_lines, None)),
            ]
            if abort_signal is not None:
                abort_task = asyncio.create_task(_watch_abort())

            _, pending = await asyncio.wait(reader_tasks, timeout=timeout)

            readers_done = len(pending) == 0

            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            if not readers_done and not aborted:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                return ExecutionResult.error_result(
                    error=f"Code timed out after {timeout}s",
                    exit_code=-1,
                    output=self._truncate("".join(stdout_lines)),
                    duration_ms=timer.elapsed_ms(),
                )

            await process.wait()

            if aborted:
                return ExecutionResult.error_result(
                    error="Aborted",
                    exit_code=-2,
                    output=self._truncate("".join(stdout_lines)),
                    duration_ms=timer.elapsed_ms(),
                )

            output = self._truncate("".join(stdout_lines))
            error_output = "".join(stderr_lines)

            if process.returncode == 0:
                return ExecutionResult.success_result(
                    output=output,
                    duration_ms=timer.elapsed_ms(),
                )
            else:
                return ExecutionResult.error_result(
                    error=error_output or f"Code failed with exit code {process.returncode}",
                    exit_code=process.returncode or 1,
                    output=output,
                    duration_ms=timer.elapsed_ms(),
                )

        except Exception as e:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return ExecutionResult.error_result(
                error=str(e),
                exit_code=-1,
                duration_ms=timer.elapsed_ms(),
            )
        finally:
            if abort_task is not None:
                if not abort_task.done():
                    abort_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await abort_task

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_namespace(self, mode: str) -> dict[str, Any]:
        """Build the execution namespace for in-process mode."""
        allowed = set(self.allowed_modules)

        def _safe_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            root = name.split(".")[0]
            if root not in allowed:
                raise ImportError(f"Import of '{name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)

        # Start with safe builtins
        safe_builtins = dict(_SAFE_BUILTINS)
        safe_builtins["__import__"] = _safe_import

        # Pre-import allowed modules as convenient top-level names
        for mod_name in self.allowed_modules:
            try:
                safe_builtins[mod_name.replace(".", "_")] = __import__(mod_name)
            except ImportError:
                pass

        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "spec": self.spec,
            "result": _UNSET,
        }

        if mode == "execute":
            namespace["ctx"] = self.ctx
        # In search mode, ctx is not available

        return namespace

    def _truncate(self, text: str) -> str:
        """Truncate text if it exceeds max_output_size."""
        if len(text) > self.max_output_size:
            return text[: self.max_output_size] + "\n... (output truncated)"
        return text
