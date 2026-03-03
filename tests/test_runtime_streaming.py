"""Tests for runtime streaming output and abort support (Phase 3)."""

from __future__ import annotations

import asyncio
import time

import pytest

from skillkit.runtime.base import ExecutionResult
from skillkit.runtime.bash import BashRuntime

# ---------------------------------------------------------------------------
# BashRuntime basic execution
# ---------------------------------------------------------------------------


class TestBashRuntimeExecute:
    @pytest.fixture
    def runtime(self) -> BashRuntime:
        return BashRuntime(default_timeout=10.0)

    @pytest.mark.asyncio
    async def test_simple_command(self, runtime: BashRuntime) -> None:
        result = await runtime.execute("echo hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_failing_command(self, runtime: BashRuntime) -> None:
        result = await runtime.execute("exit 42")
        assert not result.success
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_command_with_cwd(self, runtime: BashRuntime) -> None:
        result = await runtime.execute("pwd", cwd="/tmp")
        assert result.success
        # macOS may resolve /tmp to /private/tmp
        assert "tmp" in result.output

    @pytest.mark.asyncio
    async def test_command_with_env(self, runtime: BashRuntime) -> None:
        result = await runtime.execute(
            "echo $MY_TEST_VAR", env={"MY_TEST_VAR": "test123"}
        )
        assert result.success
        assert "test123" in result.output

    @pytest.mark.asyncio
    async def test_command_timeout(self) -> None:
        runtime = BashRuntime(default_timeout=0.5)
        result = await runtime.execute("sleep 10")
        assert not result.success
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_duration_ms_populated(self, runtime: BashRuntime) -> None:
        result = await runtime.execute("echo fast")
        assert result.duration_ms > 0


# ---------------------------------------------------------------------------
# Streaming output (on_output callback)
# ---------------------------------------------------------------------------


class TestBashRuntimeStreaming:
    @pytest.fixture
    def runtime(self) -> BashRuntime:
        return BashRuntime(default_timeout=10.0)

    @pytest.mark.asyncio
    async def test_on_output_receives_lines(self, runtime: BashRuntime) -> None:
        lines: list[str] = []
        result = await runtime.execute(
            "echo line1 && echo line2 && echo line3",
            on_output=lambda line: lines.append(line),
        )
        assert result.success
        # Each line should be received via callback
        assert len(lines) >= 3
        output = "".join(lines)
        assert "line1" in output
        assert "line2" in output
        assert "line3" in output

    @pytest.mark.asyncio
    async def test_on_output_none_uses_fast_path(self, runtime: BashRuntime) -> None:
        """Without on_output, falls back to communicate() fast path."""
        result = await runtime.execute("echo fast")
        assert result.success
        assert "fast" in result.output

    @pytest.mark.asyncio
    async def test_streaming_stderr_not_in_callback(self, runtime: BashRuntime) -> None:
        """on_output should receive stdout only, not stderr."""
        lines: list[str] = []
        await runtime.execute(
            "echo stdout_msg && echo stderr_msg >&2",
            on_output=lambda line: lines.append(line),
        )
        output = "".join(lines)
        assert "stdout_msg" in output
        assert "stderr_msg" not in output

    @pytest.mark.asyncio
    async def test_streaming_captures_full_output(self, runtime: BashRuntime) -> None:
        """result.output should contain the full stdout even when streaming."""
        lines: list[str] = []
        result = await runtime.execute(
            "echo hello_world",
            on_output=lambda line: lines.append(line),
        )
        assert "hello_world" in result.output

    @pytest.mark.asyncio
    async def test_streaming_with_failing_command(self, runtime: BashRuntime) -> None:
        lines: list[str] = []
        result = await runtime.execute(
            "echo partial && exit 1",
            on_output=lambda line: lines.append(line),
        )
        assert not result.success
        output = "".join(lines)
        assert "partial" in output


# ---------------------------------------------------------------------------
# Abort signal
# ---------------------------------------------------------------------------


class TestBashRuntimeAbort:
    @pytest.fixture
    def runtime(self) -> BashRuntime:
        return BashRuntime(default_timeout=10.0)

    @pytest.mark.asyncio
    async def test_abort_kills_process(self, runtime: BashRuntime) -> None:
        abort = asyncio.Event()

        async def set_abort_soon():
            await asyncio.sleep(0.3)
            abort.set()

        asyncio.create_task(set_abort_soon())

        result = await runtime.execute("sleep 60", abort_signal=abort)
        assert not result.success
        assert result.exit_code == -2
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_abort_with_on_output(self, runtime: BashRuntime) -> None:
        """Abort works together with streaming output."""
        abort = asyncio.Event()
        lines: list[str] = []

        async def set_abort_soon():
            await asyncio.sleep(0.3)
            abort.set()

        asyncio.create_task(set_abort_soon())

        result = await runtime.execute(
            # Print something then sleep forever
            "echo started && sleep 60",
            on_output=lambda line: lines.append(line),
            abort_signal=abort,
        )
        assert not result.success
        assert "Aborted" in (result.error or "")
        # Should have received the "started" line before abort
        output = "".join(lines)
        assert "started" in output

    @pytest.mark.asyncio
    async def test_pre_set_abort_signal(self, runtime: BashRuntime) -> None:
        """If abort signal is already set, process should be killed immediately."""
        abort = asyncio.Event()
        abort.set()  # Pre-set

        result = await runtime.execute("sleep 60", abort_signal=abort)
        assert not result.success

    @pytest.mark.asyncio
    async def test_abort_not_set_command_completes(self, runtime: BashRuntime) -> None:
        """If abort is never set, command should complete normally."""
        abort = asyncio.Event()
        result = await runtime.execute("echo normal", abort_signal=abort)
        assert result.success
        assert "normal" in result.output


class TestBashRuntimeTimingRegression:
    @pytest.mark.asyncio
    async def test_echo_with_abort_signal_finishes_quickly(self) -> None:
        runtime = BashRuntime(default_timeout=5.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await runtime.execute("echo hi", abort_signal=abort)

        elapsed = time.monotonic() - start
        assert result.success
        assert "hi" in result.output
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_sleep_two_seconds_matches_real_duration(self) -> None:
        runtime = BashRuntime(default_timeout=10.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await runtime.execute("sleep 2; echo hi", abort_signal=abort)

        elapsed = time.monotonic() - start
        assert result.success
        assert "hi" in result.output
        assert 1.7 <= elapsed <= 3.5

    @pytest.mark.asyncio
    async def test_timeout_still_applies_with_abort_signal(self) -> None:
        runtime = BashRuntime(default_timeout=10.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await runtime.execute("sleep 60", timeout=3.0, abort_signal=abort)

        elapsed = time.monotonic() - start
        assert not result.success
        assert "timed out" in (result.error or "")
        assert 2.5 <= elapsed <= 5.0

    @pytest.mark.asyncio
    async def test_abort_during_execution_returns_quickly(self) -> None:
        runtime = BashRuntime(default_timeout=10.0)
        abort = asyncio.Event()

        async def trigger_abort() -> None:
            await asyncio.sleep(0.3)
            abort.set()

        abort_task = asyncio.create_task(trigger_abort())
        start = time.monotonic()

        result = await runtime.execute("sleep 60", abort_signal=abort)
        await abort_task

        elapsed = time.monotonic() - start
        assert not result.success
        assert result.exit_code == -2
        assert "Aborted" in (result.error or "")
        assert elapsed < 2.0


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------


class TestBashRuntimeScript:
    @pytest.fixture
    def runtime(self) -> BashRuntime:
        return BashRuntime(default_timeout=10.0)

    @pytest.mark.asyncio
    async def test_execute_script_basic(self, runtime: BashRuntime) -> None:
        script = "echo line1\necho line2\nexit 0"
        result = await runtime.execute_script(script)
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_execute_script_streaming(self, runtime: BashRuntime) -> None:
        lines: list[str] = []
        script = "echo a\necho b\necho c"
        result = await runtime.execute_script(
            script, on_output=lambda line: lines.append(line)
        )
        assert result.success
        assert len(lines) >= 3

    @pytest.mark.asyncio
    async def test_execute_script_abort(self, runtime: BashRuntime) -> None:
        abort = asyncio.Event()

        async def set_abort_soon():
            await asyncio.sleep(0.3)
            abort.set()

        asyncio.create_task(set_abort_soon())

        result = await runtime.execute_script("sleep 60", abort_signal=abort)
        assert not result.success
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_script_timeout(self) -> None:
        runtime = BashRuntime(default_timeout=0.5)
        result = await runtime.execute_script("sleep 10")
        assert not result.success
        assert "timed out" in (result.error or "")


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestBashRuntimeTruncation:
    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        runtime = BashRuntime(max_output_size=50)
        # Generate output that exceeds 50 chars
        result = await runtime.execute(
            "python3 -c \"print('x' * 200)\""
        )
        assert result.success
        assert len(result.output) <= 80  # 50 + truncation message
        assert "truncated" in result.output


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_success_result(self) -> None:
        result = ExecutionResult.success_result(output="ok", duration_ms=100.0)
        assert result.success
        assert result.output == "ok"
        assert result.duration_ms == 100.0
        assert result.exit_code == 0

    def test_error_result(self) -> None:
        result = ExecutionResult.error_result(
            error="fail", exit_code=2, output="partial", duration_ms=50.0
        )
        assert not result.success
        assert result.error == "fail"
        assert result.exit_code == 2
        assert result.output == "partial"
