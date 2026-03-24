"""Tests for CodeModeRuntime (search + execute pattern)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from skillengine.runtime.code_mode import CodeModeRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0"},
    "paths": {
        "/users": {
            "get": {"summary": "List users", "tags": ["users"]},
            "post": {"summary": "Create user", "tags": ["users"]},
        },
        "/users/{id}": {
            "get": {"summary": "Get user by ID", "tags": ["users"]},
            "delete": {"summary": "Delete user", "tags": ["users"]},
        },
        "/orders": {
            "get": {"summary": "List orders", "tags": ["orders"]},
        },
        "/orders/{id}": {
            "get": {"summary": "Get order by ID", "tags": ["orders"]},
        },
    },
    "servers": [{"url": "https://api.example.com/v1"}],
}


class MockClient:
    """Fake API client for testing ctx injection."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str) -> dict:
        self.calls.append(("GET", path))
        return {"status": "ok", "path": path}

    def post(self, path: str, data: dict | None = None) -> dict:
        self.calls.append(("POST", path))
        return {"status": "created", "path": path, "data": data}


@pytest.fixture
def spec() -> dict:
    return SAMPLE_SPEC


@pytest.fixture
def client() -> MockClient:
    return MockClient()


@pytest.fixture
def runtime(spec: dict, client: MockClient) -> CodeModeRuntime:
    return CodeModeRuntime(
        spec=spec,
        ctx={"client": client},
        default_timeout=10.0,
    )


@pytest.fixture
def subprocess_runtime(spec: dict) -> CodeModeRuntime:
    return CodeModeRuntime(
        spec=spec,
        ctx={"greeting": "hello"},
        sandbox="subprocess",
        default_timeout=10.0,
    )


# ---------------------------------------------------------------------------
# In-process: search mode
# ---------------------------------------------------------------------------


class TestCodeModeSearch:
    @pytest.mark.asyncio
    async def test_search_filters_paths(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search(
            "result = [p for p in spec['paths'] if '/users' in p]"
        )
        assert result.success
        data = json.loads(result.output)
        assert "/users" in data
        assert "/users/{id}" in data
        assert "/orders" not in data

    @pytest.mark.asyncio
    async def test_search_extracts_endpoints(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search(
            """
endpoints = []
for path, methods in spec['paths'].items():
    for method, op in methods.items():
        if 'orders' in op.get('tags', []):
            endpoints.append({'method': method.upper(), 'path': path, 'summary': op['summary']})
result = endpoints
"""
        )
        assert result.success
        data = json.loads(result.output)
        assert len(data) == 2
        summaries = {ep["summary"] for ep in data}
        assert "List orders" in summaries
        assert "Get order by ID" in summaries

    @pytest.mark.asyncio
    async def test_search_print_output(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search("print(spec['info']['title'])")
        assert result.success
        assert "Test API" in result.output

    @pytest.mark.asyncio
    async def test_search_no_ctx_available(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search("result = 'ctx' in dir()")
        assert result.success
        assert "false" in result.output.lower()

    @pytest.mark.asyncio
    async def test_search_with_no_spec(self) -> None:
        rt = CodeModeRuntime(spec=None)
        result = await rt.search("result = str(spec)")
        assert result.success
        assert "None" in result.output

    @pytest.mark.asyncio
    async def test_search_error_handling(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search("result = spec['nonexistent']")
        assert not result.success
        assert "KeyError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_search_result_int(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.search("result = len(spec['paths'])")
        assert result.success
        assert "4" in result.output


# ---------------------------------------------------------------------------
# In-process: execute mode (run)
# ---------------------------------------------------------------------------


class TestCodeModeExecute:
    @pytest.mark.asyncio
    async def test_execute_with_client(
        self, runtime: CodeModeRuntime, client: MockClient
    ) -> None:
        result = await runtime.run(
            "result = ctx['client'].get('/users')"
        )
        assert result.success
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["path"] == "/users"
        assert len(client.calls) == 1
        assert client.calls[0] == ("GET", "/users")

    @pytest.mark.asyncio
    async def test_execute_uses_spec_and_ctx(
        self, runtime: CodeModeRuntime, client: MockClient
    ) -> None:
        result = await runtime.run(
            """
server = spec['servers'][0]['url']
resp = ctx['client'].get(server + '/users')
result = {'server': server, 'response': resp}
"""
        )
        assert result.success
        data = json.loads(result.output)
        assert data["server"] == "https://api.example.com/v1"
        assert data["response"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_multi_step(
        self, runtime: CodeModeRuntime, client: MockClient
    ) -> None:
        result = await runtime.run(
            """
paths = [p for p in spec['paths'] if '/users' in p]
responses = []
for path in paths:
    resp = ctx['client'].get(path)
    responses.append(resp)
result = responses
"""
        )
        assert result.success
        data = json.loads(result.output)
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_execute_error_in_code(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run("result = 1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in (result.error or "")


# ---------------------------------------------------------------------------
# Base class interface (execute / execute_script)
# ---------------------------------------------------------------------------


class TestCodeModeBaseInterface:
    @pytest.mark.asyncio
    async def test_execute_maps_to_run(
        self, runtime: CodeModeRuntime, client: MockClient
    ) -> None:
        result = await runtime.execute("result = ctx['client'].get('/test')")
        assert result.success
        data = json.loads(result.output)
        assert data["path"] == "/test"

    @pytest.mark.asyncio
    async def test_execute_script_maps_to_run(
        self, runtime: CodeModeRuntime, client: MockClient
    ) -> None:
        result = await runtime.execute_script(
            "resp = ctx['client'].post('/items', {'name': 'widget'})\nresult = resp"
        )
        assert result.success
        data = json.loads(result.output)
        assert data["status"] == "created"

    @pytest.mark.asyncio
    async def test_duration_ms_populated(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.execute("result = 42")
        assert result.success
        assert result.duration_ms > 0


# ---------------------------------------------------------------------------
# Allowed builtins and modules
# ---------------------------------------------------------------------------


class TestCodeModeSafety:
    @pytest.mark.asyncio
    async def test_json_module_available(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run("result = json.loads('{\"a\": 1}')")
        assert result.success
        data = json.loads(result.output)
        assert data["a"] == 1

    @pytest.mark.asyncio
    async def test_re_module_available(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run(
            "import re\nresult = bool(re.match(r'/users', '/users/123'))"
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_no_os_module(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run("import os\nresult = os.listdir('.')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_subprocess_module(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run("import subprocess\nresult = 'pwned'")
        assert not result.success

    @pytest.mark.asyncio
    async def test_builtins_available(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run(
            "result = sorted([3, 1, 2])"
        )
        assert result.success
        assert json.loads(result.output) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_list_comprehension(self, runtime: CodeModeRuntime) -> None:
        result = await runtime.run(
            "result = [x * 2 for x in range(5)]"
        )
        assert result.success
        assert json.loads(result.output) == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_custom_allowed_modules(self) -> None:
        rt = CodeModeRuntime(
            allowed_modules=["json", "os"],
        )
        result = await rt.run("import os\nresult = os.sep")
        assert result.success


# ---------------------------------------------------------------------------
# Subprocess mode
# ---------------------------------------------------------------------------


class TestCodeModeSubprocess:
    @pytest.mark.asyncio
    async def test_subprocess_search(self, subprocess_runtime: CodeModeRuntime) -> None:
        result = await subprocess_runtime.search(
            "result = [p for p in spec['paths'] if '/users' in p]"
        )
        assert result.success
        data = json.loads(result.output)
        assert "/users" in data

    @pytest.mark.asyncio
    async def test_subprocess_execute(self, subprocess_runtime: CodeModeRuntime) -> None:
        result = await subprocess_runtime.execute(
            "result = {'greeting': ctx['greeting'], 'count': len(spec['paths'])}"
        )
        assert result.success
        data = json.loads(result.output)
        assert data["greeting"] == "hello"
        assert data["count"] == 4

    @pytest.mark.asyncio
    async def test_subprocess_print_output(self, subprocess_runtime: CodeModeRuntime) -> None:
        result = await subprocess_runtime.execute("print('hello from subprocess')")
        assert result.success
        assert "hello from subprocess" in result.output

    @pytest.mark.asyncio
    async def test_subprocess_error(self, subprocess_runtime: CodeModeRuntime) -> None:
        result = await subprocess_runtime.execute("raise ValueError('boom')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_subprocess_timeout(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=1.0)
        result = await rt.execute("import time; time.sleep(30)")
        assert not result.success
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_subprocess_with_env(self, subprocess_runtime: CodeModeRuntime) -> None:
        result = await subprocess_runtime.execute(
            "import os; result = os.environ.get('TEST_CODE_MODE', 'missing')",
            env={"TEST_CODE_MODE": "found"},
        )
        assert result.success
        assert "found" in result.output

    @pytest.mark.asyncio
    async def test_subprocess_streaming(self, subprocess_runtime: CodeModeRuntime) -> None:
        lines: list[str] = []
        result = await subprocess_runtime.execute(
            "print('line1')\nprint('line2')\nprint('line3')",
            on_output=lambda line: lines.append(line),
        )
        assert result.success
        assert len(lines) >= 3
        output = "".join(lines)
        assert "line1" in output
        assert "line2" in output


# ---------------------------------------------------------------------------
# Abort signal
# ---------------------------------------------------------------------------


class TestCodeModeAbort:
    @pytest.mark.asyncio
    async def test_pre_set_abort(self, runtime: CodeModeRuntime) -> None:
        abort = asyncio.Event()
        abort.set()
        result = await runtime.execute("result = 'should not run'", abort_signal=abort)
        assert not result.success
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_subprocess_pre_set_abort(self, subprocess_runtime: CodeModeRuntime) -> None:
        abort = asyncio.Event()
        abort.set()
        result = await subprocess_runtime.execute(
            "result = 'should not run'", abort_signal=abort
        )
        assert not result.success
        assert "Aborted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_subprocess_abort_during_execution(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=10.0)
        abort = asyncio.Event()

        async def set_abort_soon():
            await asyncio.sleep(0.3)
            abort.set()

        asyncio.create_task(set_abort_soon())
        result = await rt.execute(
            "import time; time.sleep(60)", abort_signal=abort
        )
        assert not result.success
        assert "Aborted" in (result.error or "")


class TestCodeModeTimingRegression:
    @pytest.mark.asyncio
    async def test_subprocess_quick_code_with_abort_signal_finishes_quickly(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=5.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await rt.execute("print('hi')", abort_signal=abort)

        elapsed = time.monotonic() - start
        assert result.success
        assert "hi" in result.output
        assert elapsed < 1.5

    @pytest.mark.asyncio
    async def test_subprocess_sleep_two_seconds_matches_real_duration(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=10.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await rt.execute(
            "import time; time.sleep(2); print('hi')",
            abort_signal=abort,
        )

        elapsed = time.monotonic() - start
        assert result.success
        assert "hi" in result.output
        assert 1.7 <= elapsed <= 4.0

    @pytest.mark.asyncio
    async def test_subprocess_timeout_still_applies_with_abort_signal(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=10.0)
        abort = asyncio.Event()
        start = time.monotonic()

        result = await rt.execute(
            "import time; time.sleep(60)",
            timeout=3.0,
            abort_signal=abort,
        )

        elapsed = time.monotonic() - start
        assert not result.success
        assert "timed out" in (result.error or "")
        assert 2.5 <= elapsed <= 5.5

    @pytest.mark.asyncio
    async def test_subprocess_abort_returns_quickly(self) -> None:
        rt = CodeModeRuntime(sandbox="subprocess", default_timeout=10.0)
        abort = asyncio.Event()

        async def set_abort_soon() -> None:
            await asyncio.sleep(0.3)
            abort.set()

        abort_task = asyncio.create_task(set_abort_soon())
        start = time.monotonic()

        result = await rt.execute(
            "import time; time.sleep(60)",
            abort_signal=abort,
        )
        await abort_task

        elapsed = time.monotonic() - start
        assert not result.success
        assert result.exit_code == -2
        assert "Aborted" in (result.error or "")
        assert elapsed < 2.5


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestCodeModeTruncation:
    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        rt = CodeModeRuntime(max_output_size=50)
        result = await rt.execute("print('x' * 200)")
        assert result.success
        assert len(result.output) <= 80  # 50 + truncation message
        assert "truncated" in result.output


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class TestCodeModeToolDefinitions:
    def test_tool_definitions_with_spec(self, runtime: CodeModeRuntime) -> None:
        tools = runtime.get_tool_definitions()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"search", "execute"}

    def test_tool_definitions_without_spec(self) -> None:
        rt = CodeModeRuntime(ctx={"client": "mock"})
        tools = rt.get_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "execute"

    def test_search_tool_includes_spec_hint(self, runtime: CodeModeRuntime) -> None:
        tools = runtime.get_tool_definitions()
        search_tool = next(t for t in tools if t["function"]["name"] == "search")
        desc = search_tool["function"]["description"]
        assert "Top-level keys" in desc

    def test_execute_tool_lists_ctx_keys(self, runtime: CodeModeRuntime) -> None:
        tools = runtime.get_tool_definitions()
        exec_tool = next(t for t in tools if t["function"]["name"] == "execute")
        desc = exec_tool["function"]["description"]
        assert "client" in desc

    def test_tool_definitions_code_parameter(self, runtime: CodeModeRuntime) -> None:
        tools = runtime.get_tool_definitions()
        for tool in tools:
            params = tool["function"]["parameters"]
            assert "code" in params["properties"]
            assert "code" in params["required"]


# ---------------------------------------------------------------------------
# Integration: engine-level usage
# ---------------------------------------------------------------------------


class TestCodeModeEngineIntegration:
    @pytest.mark.asyncio
    async def test_used_as_engine_runtime(self, spec: dict, client: MockClient) -> None:
        """CodeModeRuntime can be passed to SkillsEngine as runtime."""
        from skillengine.config import SkillsConfig
        from skillengine.engine import SkillsEngine

        runtime = CodeModeRuntime(spec=spec, ctx={"client": client})
        engine = SkillsEngine(
            config=SkillsConfig(skill_dirs=[]),
            runtime=runtime,
        )

        # Engine.execute delegates to CodeModeRuntime
        result = await engine.execute(
            "result = [p for p in spec['paths'] if '/orders' in p]"
        )
        assert result.success
        data = json.loads(result.output)
        assert "/orders" in data
