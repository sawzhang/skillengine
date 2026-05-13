"""Workflow executor — walks the DAG and produces a final context."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from skillengine.workflow.models import (
    BranchNode,
    NodeResult,
    Workflow,
    WorkflowContext,
)


class WorkflowError(RuntimeError):
    """Raised when a workflow fails or is malformed."""


CheckpointSink = Callable[[str, str, WorkflowContext], None]


class WorkflowExecutor:
    """Walk a :class:`Workflow` from its start node to a terminal.

    The executor is intentionally minimal — node implementations are the only
    place where domain knowledge lives.  Plug in agents and tools via the
    ``agents`` / ``tools`` factory maps; plug in a durable persistence layer
    via ``checkpoint_sink`` (used by :class:`CheckpointNode`).
    """

    def __init__(
        self,
        workflow: Workflow,
        agents: dict[str, Any] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        checkpoint_sink: CheckpointSink | None = None,
        max_steps: int = 1000,
    ) -> None:
        self.workflow = workflow
        self._agents: dict[str, Any] = agents or {}
        self._tools: dict[str, Callable[..., Any]] = tools or {}
        self.checkpoint_sink: CheckpointSink | None = checkpoint_sink
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, initial_state: dict[str, Any] | None = None) -> WorkflowContext:
        ctx = WorkflowContext(state=dict(initial_state or {}))
        if not self.workflow.start:
            raise WorkflowError("workflow has no start node")
        await self._run_from(self.workflow.start, ctx)
        return ctx

    async def resume(self, ctx: WorkflowContext) -> WorkflowContext:
        """Continue execution from ``ctx.current_node``.

        Used by FLOW-2 durable resume.  If ``current_node`` is None, behaves
        like a fresh run from :attr:`Workflow.start`.
        """
        start = ctx.current_node or self.workflow.start
        if not start:
            raise WorkflowError("workflow has no start node and ctx has no current_node")
        await self._run_from(start, ctx)
        return ctx

    def get_agent(self, agent_id: str) -> Any:
        if agent_id not in self._agents:
            raise WorkflowError(f"agent factory {agent_id!r} not registered")
        return self._agents[agent_id]

    def get_tool(self, tool_id: str) -> Callable[..., Any]:
        if tool_id not in self._tools:
            raise WorkflowError(f"tool {tool_id!r} not registered")
        return self._tools[tool_id]

    def register_agent(self, agent_id: str, agent: Any) -> None:
        self._agents[agent_id] = agent

    def register_tool(self, tool_id: str, tool: Callable[..., Any]) -> None:
        self._tools[tool_id] = tool

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_from(self, start_id: str, ctx: WorkflowContext) -> Any:
        current_id: str | None = start_id
        steps = 0
        last_output: Any = None
        while current_id is not None:
            if steps >= self.max_steps:
                raise WorkflowError(f"workflow exceeded max_steps={self.max_steps}")
            if current_id not in self.workflow.nodes:
                raise WorkflowError(f"unknown node id: {current_id!r}")
            node = self.workflow.nodes[current_id]
            ctx.current_node = current_id

            started = time.perf_counter()
            error: str | None = None
            output: Any = None
            try:
                output = await node.run(ctx, self)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                duration_ms = (time.perf_counter() - started) * 1000
                ctx.record(
                    NodeResult(
                        node_id=node.id,
                        node_type=node.node_type,
                        output=None,
                        error=error,
                        duration_ms=duration_ms,
                    )
                )
                raise WorkflowError(f"node {node.id!r} failed: {error}") from exc

            duration_ms = (time.perf_counter() - started) * 1000
            ctx.record(
                NodeResult(
                    node_id=node.id,
                    node_type=node.node_type,
                    output=output,
                    duration_ms=duration_ms,
                )
            )
            last_output = output

            # Branch nodes choose their own successor.
            if isinstance(node, BranchNode):
                current_id = output if isinstance(output, str) else node.next
            else:
                current_id = node.next
            steps += 1

        ctx.current_node = None
        return last_output
