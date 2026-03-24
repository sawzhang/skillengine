"""SandboxedAgentRunner — all tool operations execute inside a BoxLite sandbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.logging import get_logger
from skillengine.runtime.boxlite import BoxLiteRuntime

logger = get_logger(__name__)


class SandboxedAgentRunner(AgentRunner):
    """AgentRunner that redirects file and execution tools into a BoxLite sandbox.

    - ``execute`` / ``execute_script``: routed through the engine runtime
      (which should be a :class:`BoxLiteRuntime`).
    - ``read``: delegated to :meth:`BoxLiteRuntime.read_file`.
    - ``write``: delegated to :meth:`BoxLiteRuntime.write_file`.
    - ``skill`` and other tools: handled by the parent class.
    """

    def __init__(
        self,
        engine: SkillsEngine,
        config: AgentConfig | None = None,
        box_runtime: BoxLiteRuntime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(engine=engine, config=config, **kwargs)
        self._box_runtime: BoxLiteRuntime = box_runtime or self._resolve_box_runtime(engine)

    @staticmethod
    def _resolve_box_runtime(engine: SkillsEngine) -> BoxLiteRuntime:
        """Extract a BoxLiteRuntime from the engine, or raise."""
        rt = engine.runtime
        if not isinstance(rt, BoxLiteRuntime):
            raise TypeError(
                f"SandboxedAgentRunner requires a BoxLiteRuntime, got {type(rt).__name__}. "
                "Pass box_runtime explicitly or use BoxLiteRuntime as the engine runtime."
            )
        return rt

    # ------------------------------------------------------------------
    # Tool dispatch override
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_call: dict[str, Any],
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        """Dispatch tool calls, redirecting read/write into the sandbox."""
        name = tool_call.get("name", "")
        args_str = tool_call.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {args_str}"

        # ---- read: sandbox file read ----
        if name == "read":
            file_path = args.get("path", "")
            try:
                text = await self._box_runtime.read_file(file_path)
                if len(text) > 100_000:
                    text = text[:100_000] + "\n... (truncated)"
                return text
            except FileNotFoundError:
                return f"Error: File not found: {file_path}"
            except Exception as e:
                return f"Error reading file: {e}"

        # ---- write: sandbox file write ----
        if name == "write":
            file_path = args.get("path", "")
            content = args.get("content", "")
            try:
                await self._box_runtime.write_file(file_path, content)
                return f"Written {len(content)} bytes to {file_path}"
            except Exception as e:
                return f"Error writing file: {e}"

        # ---- everything else: parent handles (execute/execute_script go
        #      through the engine → BoxLiteRuntime, skill tool, etc.) ----
        return await super()._execute_tool(tool_call, on_output)
