"""Tests for SandboxedAgentRunner."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the boxlite SDK before importing SandboxedAgentRunner
# ---------------------------------------------------------------------------

if "boxlite" not in sys.modules:
    _mock_boxlite = types.ModuleType("boxlite")
    _mock_boxlite.Boxlite = MagicMock  # type: ignore[attr-defined]
    _mock_boxlite.Box = MagicMock  # type: ignore[attr-defined]
    _mock_boxlite.BoxOptions = MagicMock  # type: ignore[attr-defined]
    _mock_boxlite.ExecResult = MagicMock  # type: ignore[attr-defined]
    _mock_boxlite.Execution = MagicMock  # type: ignore[attr-defined]
    sys.modules["boxlite"] = _mock_boxlite

from skillengine.config import SkillsConfig  # noqa: E402
from skillengine.engine import SkillsEngine  # noqa: E402
from skillengine.runtime.boxlite import BoxLiteRuntime  # noqa: E402
from skillengine.sandbox.runner import SandboxedAgentRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exit_code = exit_code
    return result


def _make_execution(exec_result: MagicMock | None = None) -> AsyncMock:
    if exec_result is None:
        exec_result = _make_exec_result(stdout="ok\n")
    execution = AsyncMock()
    execution.wait = AsyncMock(return_value=exec_result)
    execution.kill = AsyncMock()
    return execution


def _make_box(execution: AsyncMock | None = None) -> AsyncMock:
    if execution is None:
        execution = _make_execution()
    box = AsyncMock()
    box.exec = AsyncMock(return_value=execution)
    box.stop = AsyncMock()
    box.remove = AsyncMock()
    return box


def _make_runner(mock_box: AsyncMock | None = None) -> tuple[SandboxedAgentRunner, BoxLiteRuntime]:
    """Create a SandboxedAgentRunner with a mocked BoxLiteRuntime."""
    if mock_box is None:
        mock_box = _make_box()
    runtime = BoxLiteRuntime(default_timeout=10.0)
    runtime._box = mock_box
    engine = SkillsEngine(config=SkillsConfig(skill_dirs=[]), runtime=runtime)
    runner = SandboxedAgentRunner(engine=engine, box_runtime=runtime)
    return runner, runtime


# ---------------------------------------------------------------------------
# Read tool redirection
# ---------------------------------------------------------------------------


class TestSandboxedRunnerRead:
    @pytest.mark.asyncio
    async def test_read_redirects_to_sandbox(self) -> None:
        mock_box = _make_box()
        exec_result = _make_exec_result(stdout="hello world\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "read",
            "arguments": json.dumps({"path": "/workspace/file.txt"}),
            "id": "call_1",
        }
        result = await runner._execute_tool(tool_call)
        assert "hello world" in result
        mock_box.exec.assert_called_once_with("cat", "/workspace/file.txt")

    @pytest.mark.asyncio
    async def test_read_file_not_found(self) -> None:
        mock_box = _make_box()
        exec_result = _make_exec_result(stderr="No such file", exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "read",
            "arguments": json.dumps({"path": "/nonexistent"}),
            "id": "call_2",
        }
        result = await runner._execute_tool(tool_call)
        assert "Error: File not found" in result

    @pytest.mark.asyncio
    async def test_read_truncates_large_output(self) -> None:
        mock_box = _make_box()
        large_content = "x" * 200_000
        exec_result = _make_exec_result(stdout=large_content)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "read",
            "arguments": json.dumps({"path": "/workspace/big.txt"}),
            "id": "call_3",
        }
        result = await runner._execute_tool(tool_call)
        assert len(result) <= 100_020  # 100k + truncation message
        assert "truncated" in result


# ---------------------------------------------------------------------------
# Write tool redirection
# ---------------------------------------------------------------------------


class TestSandboxedRunnerWrite:
    @pytest.mark.asyncio
    async def test_write_redirects_to_sandbox(self) -> None:
        mock_box = _make_box()
        exec_result = _make_exec_result()
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "write",
            "arguments": json.dumps({"path": "/workspace/out.txt", "content": "data"}),
            "id": "call_4",
        }
        result = await runner._execute_tool(tool_call)
        assert "Written 4 bytes" in result
        # Should have called bash -c with write command
        call_args = mock_box.exec.call_args
        assert call_args[0][0] == "bash"
        assert "/workspace/out.txt" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_write_failure(self) -> None:
        mock_box = _make_box()
        exec_result = _make_exec_result(stderr="Read-only", exit_code=1)
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "write",
            "arguments": json.dumps({"path": "/ro/file.txt", "content": "data"}),
            "id": "call_5",
        }
        result = await runner._execute_tool(tool_call)
        assert "Error writing file" in result


# ---------------------------------------------------------------------------
# Execute tool still goes through engine (sandbox runtime)
# ---------------------------------------------------------------------------


class TestSandboxedRunnerExecute:
    @pytest.mark.asyncio
    async def test_execute_goes_through_parent(self) -> None:
        mock_box = _make_box()
        exec_result = _make_exec_result(stdout="cmd output\n")
        execution = _make_execution(exec_result)
        mock_box.exec = AsyncMock(return_value=execution)

        runner, _ = _make_runner(mock_box)
        tool_call = {
            "name": "execute",
            "arguments": json.dumps({"command": "echo hello"}),
            "id": "call_6",
        }
        result = await runner._execute_tool(tool_call)
        assert "cmd output" in result


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestSandboxedRunnerInit:
    def test_explicit_box_runtime(self) -> None:
        runtime = BoxLiteRuntime()
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[]), runtime=runtime)
        runner = SandboxedAgentRunner(engine=engine, box_runtime=runtime)
        assert runner._box_runtime is runtime

    def test_infers_box_runtime_from_engine(self) -> None:
        runtime = BoxLiteRuntime()
        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[]), runtime=runtime)
        runner = SandboxedAgentRunner(engine=engine)
        assert runner._box_runtime is runtime

    def test_rejects_non_boxlite_runtime(self) -> None:
        from skillengine.runtime.bash import BashRuntime

        engine = SkillsEngine(config=SkillsConfig(skill_dirs=[]), runtime=BashRuntime())
        with pytest.raises(TypeError, match="requires a BoxLiteRuntime"):
            SandboxedAgentRunner(engine=engine)


# ---------------------------------------------------------------------------
# Invalid JSON arguments
# ---------------------------------------------------------------------------


class TestSandboxedRunnerEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self) -> None:
        runner, _ = _make_runner()
        tool_call = {
            "name": "read",
            "arguments": "not valid json",
            "id": "call_7",
        }
        result = await runner._execute_tool(tool_call)
        assert "Error: Invalid JSON arguments" in result
