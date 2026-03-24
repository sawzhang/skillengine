"""Tests for BoxLiteRuntime (VM-level sandbox execution)."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the boxlite SDK before importing BoxLiteRuntime
# ---------------------------------------------------------------------------

_mock_boxlite = types.ModuleType("boxlite")
_mock_boxlite.Boxlite = MagicMock  # type: ignore[attr-defined]
_mock_boxlite.Box = MagicMock  # type: ignore[attr-defined]
_mock_boxlite.BoxOptions = MagicMock  # type: ignore[attr-defined]
_mock_boxlite.ExecResult = MagicMock  # type: ignore[attr-defined]
_mock_boxlite.Execution = MagicMock  # type: ignore[attr-defined]

sys.modules["boxlite"] = _mock_boxlite

from skillengine.runtime.base import ExecutionResult  # noqa: E402
from skillengine.runtime.boxlite import BoxLiteRuntime, SecurityLevel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> MagicMock:
    """Create a mock ExecResult."""
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exit_code = exit_code
    return result


def _make_execution(exec_result: MagicMock | None = None) -> AsyncMock:
    """Create a mock Execution that returns the given ExecResult on wait()."""
    if exec_result is None:
        exec_result = _make_exec_result(stdout="ok\n")
    execution = AsyncMock()
    execution.wait = AsyncMock(return_value=exec_result)
    execution.kill = AsyncMock()
    return execution


def _make_box(execution: AsyncMock | None = None) -> AsyncMock:
    """Create a mock Box that returns the given Execution on exec()."""
    if execution is None:
        execution = _make_execution()
    box = AsyncMock()
    box.exec = AsyncMock(return_value=execution)
    box.stop = AsyncMock()
    box.remove = AsyncMock()
    return box


def _make_boxlite_runtime_obj(box: AsyncMock | None = None) -> AsyncMock:
    """Create a mock Boxlite runtime that produces the given Box."""
    if box is None:
        box = _make_box()
    bl_runtime = AsyncMock()
    bl_runtime.create = AsyncMock(return_value=box)
    return bl_runtime


@pytest.fixture
def mock_box() -> AsyncMock:
    return _make_box()


@pytest.fixture
def runtime(mock_box: AsyncMock) -> BoxLiteRuntime:
    """Pre-configured BoxLiteRuntime with mocked box."""
    rt = BoxLiteRuntime(default_timeout=10.0)
    # Inject the mock box directly, bypassing lazy init
    rt._box = mock_box
    return rt


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeExecute:
    @pytest.mark.asyncio
    async def test_simple_command(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="hello\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("echo hello")
        assert result.success
        assert "hello" in result.output
        mock_box.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_failing_command(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="", stderr="not found", exit_code=127)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("nonexistent_cmd")
        assert not result.success
        assert result.exit_code == 127
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_command_with_cwd(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="/tmp\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("pwd", cwd="/tmp")
        assert result.success
        # The command should include "cd /tmp"
        call_args = mock_box.exec.call_args
        cmd_str = call_args[0][2]  # "bash", "-c", <command>
        assert "cd /tmp" in cmd_str

    @pytest.mark.asyncio
    async def test_command_with_env(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="test123\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("echo $MY_VAR", env={"MY_VAR": "test123"})
        assert result.success
        call_args = mock_box.exec.call_args
        cmd_str = call_args[0][2]
        assert "export MY_VAR='test123'" in cmd_str

    @pytest.mark.asyncio
    async def test_duration_ms_populated(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="fast\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("echo fast")
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_exception_returns_error_result(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        mock_box.exec = AsyncMock(side_effect=RuntimeError("connection failed"))

        result = await runtime.execute("echo hello")
        assert not result.success
        assert "connection failed" in (result.error or "")
        assert result.exit_code == -1


# ---------------------------------------------------------------------------
# Execute script
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeScript:
    @pytest.mark.asyncio
    async def test_execute_script_basic(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        # write_file exec, chmod exec, run exec, cleanup exec
        write_exec = _make_execution(_make_exec_result())
        chmod_exec = _make_execution(_make_exec_result())
        run_exec = _make_execution(_make_exec_result(stdout="line1\nline2\n"))
        cleanup_exec = _make_execution(_make_exec_result())

        mock_box.exec = AsyncMock(
            side_effect=[write_exec, chmod_exec, run_exec, cleanup_exec]
        )

        result = await runtime.execute_script("echo line1\necho line2")
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_execute_script_failure(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        write_exec = _make_execution(_make_exec_result())
        chmod_exec = _make_execution(_make_exec_result())
        run_exec = _make_execution(_make_exec_result(stderr="error", exit_code=1))
        cleanup_exec = _make_execution(_make_exec_result())

        mock_box.exec = AsyncMock(
            side_effect=[write_exec, chmod_exec, run_exec, cleanup_exec]
        )

        result = await runtime.execute_script("exit 1")
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_script_writes_temp_file(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        write_exec = _make_execution(_make_exec_result())
        chmod_exec = _make_execution(_make_exec_result())
        run_exec = _make_execution(_make_exec_result(stdout="ok\n"))
        cleanup_exec = _make_execution(_make_exec_result())

        mock_box.exec = AsyncMock(
            side_effect=[write_exec, chmod_exec, run_exec, cleanup_exec]
        )

        await runtime.execute_script("echo ok")

        # First call should write the script file
        first_call = mock_box.exec.call_args_list[0]
        cmd_str = first_call[0][2]  # "bash", "-c", <write command>
        assert "cat >" in cmd_str
        assert "_skillengine_" in cmd_str
        assert "SKILLKIT_SCRIPT_EOF" in cmd_str


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeStreaming:
    @pytest.mark.asyncio
    async def test_on_output_receives_lines(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="line1\nline2\nline3\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        lines: list[str] = []
        result = await runtime.execute(
            "echo line1 && echo line2 && echo line3",
            on_output=lambda line: lines.append(line),
        )
        assert result.success
        assert len(lines) >= 3
        output = "".join(lines)
        assert "line1" in output
        assert "line2" in output
        assert "line3" in output

    @pytest.mark.asyncio
    async def test_on_output_none_still_works(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="fast\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("echo fast")
        assert result.success
        assert "fast" in result.output

    @pytest.mark.asyncio
    async def test_streaming_captures_full_output(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="hello_world\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        lines: list[str] = []
        result = await runtime.execute(
            "echo hello_world",
            on_output=lambda line: lines.append(line),
        )
        assert "hello_world" in result.output

    @pytest.mark.asyncio
    async def test_streaming_with_failing_command(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="partial\n", stderr="error", exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

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


class TestBoxLiteRuntimeAbort:
    @pytest.mark.asyncio
    async def test_pre_set_abort_signal(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        execution = _make_execution(_make_exec_result())
        mock_box.exec = AsyncMock(return_value=execution)

        abort = asyncio.Event()
        abort.set()

        result = await runtime.execute("sleep 60", abort_signal=abort)
        assert not result.success
        assert result.exit_code == -2
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_abort_during_execution(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        # Make execution.wait() block until we cancel it
        never_done: asyncio.Future[MagicMock] = asyncio.get_event_loop().create_future()
        execution = AsyncMock()
        execution.wait = AsyncMock(return_value=never_done)
        execution.kill = AsyncMock()

        # Override wait to actually await the future (so it blocks)
        async def slow_wait():
            return await never_done

        execution.wait = slow_wait
        mock_box.exec = AsyncMock(return_value=execution)

        abort = asyncio.Event()

        async def set_abort_soon():
            await asyncio.sleep(0.1)
            abort.set()

        asyncio.create_task(set_abort_soon())

        result = await runtime.execute("sleep 60", abort_signal=abort)
        assert not result.success
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_abort_not_set_command_completes(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="normal\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        abort = asyncio.Event()
        result = await runtime.execute("echo normal", abort_signal=abort)
        assert result.success
        assert "normal" in result.output


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeTimeout:
    @pytest.mark.asyncio
    async def test_timeout_fires(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime(default_timeout=0.2)
        runtime._box = mock_box

        # Make execution.wait() block forever
        async def slow_wait():
            await asyncio.sleep(999)

        execution = AsyncMock()
        execution.wait = slow_wait
        execution.kill = AsyncMock()
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("sleep 60")
        assert not result.success
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_timeout_with_abort_signal(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime(default_timeout=0.2)
        runtime._box = mock_box

        async def slow_wait():
            await asyncio.sleep(999)

        execution = AsyncMock()
        execution.wait = slow_wait
        execution.kill = AsyncMock()
        mock_box.exec = AsyncMock(return_value=execution)

        abort = asyncio.Event()  # Not set — timeout should fire
        result = await runtime.execute("sleep 60", abort_signal=abort)
        assert not result.success
        assert "timed out" in (result.error or "")


# ---------------------------------------------------------------------------
# Lifecycle management
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeLifecycle:
    @pytest.mark.asyncio
    async def test_lazy_initialization(self) -> None:
        runtime = BoxLiteRuntime()
        assert runtime._box is None

    @pytest.mark.asyncio
    async def test_ensure_box_creates_box(self) -> None:
        runtime = BoxLiteRuntime()
        bl_runtime = _make_boxlite_runtime_obj()

        with patch.object(_mock_boxlite, "Boxlite") as mock_cls:
            mock_cls.default.return_value = bl_runtime
            with patch.object(_mock_boxlite, "BoxOptions", return_value=MagicMock()):
                box = await runtime._ensure_box()
                assert box is not None
                bl_runtime.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_box_reuses_existing(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime()
        runtime._box = mock_box

        box = await runtime._ensure_box()
        assert box is mock_box

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime(auto_destroy=True)
        bl_runtime = _make_boxlite_runtime_obj(mock_box)

        with patch.object(_mock_boxlite, "Boxlite") as mock_cls:
            mock_cls.default.return_value = bl_runtime
            with patch.object(_mock_boxlite, "BoxOptions", return_value=MagicMock()):
                async with runtime as rt:
                    assert rt._box is not None

        mock_box.remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_no_auto_destroy(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime(auto_destroy=False)
        bl_runtime = _make_boxlite_runtime_obj(mock_box)

        with patch.object(_mock_boxlite, "Boxlite") as mock_cls:
            mock_cls.default.return_value = bl_runtime
            with patch.object(_mock_boxlite, "BoxOptions", return_value=MagicMock()):
                async with runtime as rt:
                    assert rt._box is not None

        mock_box.remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_destroy(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime()
        runtime._box = mock_box

        await runtime.destroy()
        mock_box.remove.assert_called_once()
        assert runtime._box is None

    @pytest.mark.asyncio
    async def test_stop(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime()
        runtime._box = mock_box

        await runtime.stop()
        mock_box.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_eagerly_initializes(self) -> None:
        runtime = BoxLiteRuntime()
        bl_runtime = _make_boxlite_runtime_obj()

        with patch.object(_mock_boxlite, "Boxlite") as mock_cls:
            mock_cls.default.return_value = bl_runtime
            with patch.object(_mock_boxlite, "BoxOptions", return_value=MagicMock()):
                await runtime.start()
                assert runtime._box is not None


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeTruncation:
    @pytest.mark.asyncio
    async def test_output_truncation(self, mock_box: AsyncMock) -> None:
        runtime = BoxLiteRuntime(max_output_size=50)
        runtime._box = mock_box

        long_output = "x" * 200
        exec_result = _make_exec_result(stdout=long_output)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.execute("generate_long_output")
        assert result.success
        assert len(result.output) <= 80  # 50 + truncation message
        assert "truncated" in result.output


# ---------------------------------------------------------------------------
# Security levels
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeSecurity:
    def test_dev_security_options(self) -> None:
        runtime = BoxLiteRuntime(security_level=SecurityLevel.DEV, cpus=1, memory_mib=512)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["cpus"] >= 2
            assert call_kwargs["memory_mib"] >= 1024

    def test_standard_security_options(self) -> None:
        runtime = BoxLiteRuntime(security_level=SecurityLevel.STANDARD, cpus=1, memory_mib=512)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["cpus"] == 1
            assert call_kwargs["memory_mib"] == 512

    def test_maximum_security_options(self) -> None:
        runtime = BoxLiteRuntime(security_level=SecurityLevel.MAXIMUM, cpus=4, memory_mib=2048)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["cpus"] == 1
            assert call_kwargs["memory_mib"] == 256

    def test_security_level_enum_values(self) -> None:
        assert SecurityLevel.DEV.value == "dev"
        assert SecurityLevel.STANDARD.value == "standard"
        assert SecurityLevel.MAXIMUM.value == "maximum"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeEngineIntegration:
    @pytest.mark.asyncio
    async def test_used_as_engine_runtime(self, mock_box: AsyncMock) -> None:
        """BoxLiteRuntime can be passed to SkillsEngine as runtime."""
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        exec_result = _make_exec_result(stdout="engine_output\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runtime = BoxLiteRuntime()
        runtime._box = mock_box

        engine = SkillsEngine(
            config=SkillsConfig(skill_dirs=[]),
            runtime=runtime,
        )

        result = await engine.execute("echo engine_output")
        assert result.success
        assert "engine_output" in result.output


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeCommandBuilding:
    def test_build_command_simple(self) -> None:
        runtime = BoxLiteRuntime()
        cmd = runtime._build_command("echo hello")
        assert cmd == "echo hello"

    def test_build_command_with_cwd(self) -> None:
        runtime = BoxLiteRuntime()
        cmd = runtime._build_command("pwd", cwd="/tmp")
        assert cmd == "cd /tmp && pwd"

    def test_build_command_with_env(self) -> None:
        runtime = BoxLiteRuntime()
        cmd = runtime._build_command("echo $FOO", env={"FOO": "bar"})
        assert "export FOO='bar'" in cmd
        assert "echo $FOO" in cmd

    def test_build_command_with_cwd_and_env(self) -> None:
        runtime = BoxLiteRuntime()
        cmd = runtime._build_command("echo $FOO", cwd="/tmp", env={"FOO": "bar"})
        assert "export FOO='bar'" in cmd
        assert "cd /tmp" in cmd
        assert "echo $FOO" in cmd

    def test_build_command_env_escapes_single_quotes(self) -> None:
        runtime = BoxLiteRuntime()
        cmd = runtime._build_command("echo $V", env={"V": "it's"})
        assert "it'\\''s" in cmd


# ---------------------------------------------------------------------------
# Volumes / working_dir / box_env
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeVolumes:
    def test_volumes_passed_to_box_options(self) -> None:
        vols = [("/host/dir", "/guest/dir", "rw")]
        runtime = BoxLiteRuntime(volumes=vols)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["volumes"] == vols

    def test_working_dir_passed_to_box_options(self) -> None:
        runtime = BoxLiteRuntime(working_dir="/workspace")
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["working_dir"] == "/workspace"

    def test_box_env_passed_to_box_options(self) -> None:
        env = [("MY_VAR", "value")]
        runtime = BoxLiteRuntime(box_env=env)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["env"] == env

    def test_none_values_omitted_from_box_options(self) -> None:
        runtime = BoxLiteRuntime()
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert "volumes" not in call_kwargs
            assert "working_dir" not in call_kwargs
            assert "env" not in call_kwargs

    def test_volumes_with_dev_security(self) -> None:
        vols = [("/data", "/mnt/data", "ro")]
        runtime = BoxLiteRuntime(security_level=SecurityLevel.DEV, volumes=vols)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["volumes"] == vols
            assert call_kwargs["cpus"] >= 2

    def test_volumes_with_maximum_security(self) -> None:
        vols = [("/data", "/mnt/data", "ro")]
        runtime = BoxLiteRuntime(security_level=SecurityLevel.MAXIMUM, volumes=vols)
        with patch.object(_mock_boxlite, "BoxOptions") as mock_opts:
            runtime._resolve_box_options()
            call_kwargs = mock_opts.call_args[1]
            assert call_kwargs["volumes"] == vols
            assert call_kwargs["cpus"] == 1
            assert call_kwargs["memory_mib"] == 256


# ---------------------------------------------------------------------------
# File I/O methods
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeFileIO:
    @pytest.mark.asyncio
    async def test_read_file(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result(stdout="file contents\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        content = await runtime.read_file("/workspace/test.txt")
        assert content == "file contents\n"
        mock_box.exec.assert_called_once_with("cat", "/workspace/test.txt")

    @pytest.mark.asyncio
    async def test_read_file_not_found(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stderr="No such file", exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        with pytest.raises(FileNotFoundError, match="No such file"):
            await runtime.read_file("/nonexistent")

    @pytest.mark.asyncio
    async def test_write_file(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        exec_result = _make_exec_result()
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        await runtime.write_file("/workspace/out.txt", "hello")
        mock_box.exec.assert_called_once()
        call_args = mock_box.exec.call_args
        assert call_args[0][0] == "bash"
        assert "/workspace/out.txt" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_write_file_failure(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stderr="Permission denied", exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        with pytest.raises(OSError, match="Permission denied"):
            await runtime.write_file("/readonly/file.txt", "data")

    @pytest.mark.asyncio
    async def test_file_exists_true(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(exit_code=0)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        assert await runtime.file_exists("/workspace/test.txt") is True
        mock_box.exec.assert_called_once_with("test", "-e", "/workspace/test.txt")

    @pytest.mark.asyncio
    async def test_file_exists_false(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        assert await runtime.file_exists("/nonexistent") is False

    @pytest.mark.asyncio
    async def test_list_dir(self, runtime: BoxLiteRuntime, mock_box: AsyncMock) -> None:
        ls_output = "total 4\n-rw-r--r-- 1 root root 5 file.txt\n"
        exec_result = _make_exec_result(stdout=ls_output)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        result = await runtime.list_dir("/workspace")
        assert "file.txt" in result
        mock_box.exec.assert_called_once_with("ls", "-la", "/workspace")

    @pytest.mark.asyncio
    async def test_list_dir_not_found(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stderr="No such directory", exit_code=2)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        with pytest.raises(FileNotFoundError, match="No such directory"):
            await runtime.list_dir("/nonexistent")


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


class TestBoxLiteRuntimeIsReady:
    @pytest.mark.asyncio
    async def test_is_ready_true(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        exec_result = _make_exec_result(stdout="ready\n", exit_code=0)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        assert await runtime.is_ready() is True

    @pytest.mark.asyncio
    async def test_is_ready_no_box(self) -> None:
        runtime = BoxLiteRuntime()
        assert await runtime.is_ready() is False

    @pytest.mark.asyncio
    async def test_is_ready_exec_fails(
        self, runtime: BoxLiteRuntime, mock_box: AsyncMock
    ) -> None:
        mock_box.exec = AsyncMock(side_effect=RuntimeError("connection lost"))
        assert await runtime.is_ready() is False
