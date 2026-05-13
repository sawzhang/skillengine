"""Workflow node types and the graph container.

All nodes are JSON-serialisable via :meth:`to_dict` / :meth:`from_dict`.
A :class:`Workflow` is just a mapping of node id → node plus a start id, so
the whole graph can be checkpointed to disk and restored later.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skillengine.workflow.executor import WorkflowExecutor


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Audit-trail entry for a single node execution."""

    node_id: str
    node_type: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeResult:
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            output=data.get("output"),
            error=data.get("error"),
            duration_ms=float(data.get("duration_ms", 0.0)),
            started_at=float(data.get("started_at", time.time())),
        )


@dataclass
class WorkflowContext:
    """Mutable state threaded through a workflow run."""

    state: dict[str, Any] = field(default_factory=dict)
    history: list[NodeResult] = field(default_factory=list)
    current_node: str | None = None

    def record(self, result: NodeResult) -> None:
        self.history.append(result)
        if result.output is not None:
            self.state[result.node_id] = result.output

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "history": [h.to_dict() for h in self.history],
            "current_node": self.current_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowContext:
        return cls(
            state=dict(data.get("state", {})),
            history=[NodeResult.from_dict(h) for h in data.get("history", [])],
            current_node=data.get("current_node"),
        )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowNode(ABC):
    """Base class for every workflow node."""

    id: str
    next: str | None = None  # id of the next node, or None for terminal

    node_type: str = field(init=False, default="abstract")

    @abstractmethod
    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        """Execute this node and return its output."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.node_type, "id": self.id, "next": self.next}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowNode:
        node_type = data.get("type")
        registry = _NODE_REGISTRY
        if node_type not in registry:
            raise ValueError(f"unknown workflow node type: {node_type!r}")
        return registry[node_type]._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> WorkflowNode:
        raise NotImplementedError


@dataclass
class AgentNode(WorkflowNode):
    """Invoke an ``AgentRunner`` with a prompt rendered from context state.

    ``prompt`` may contain ``{key}`` placeholders that are filled from
    ``ctx.state`` at run time.  ``agent_factory_id`` looks up a callable in the
    executor's factory map.
    """

    prompt: str = ""
    agent_factory_id: str = "default"

    def __post_init__(self) -> None:
        self.node_type = "agent"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        rendered = _render(self.prompt, ctx.state)
        agent = executor.get_agent(self.agent_factory_id)
        response = await agent.chat(rendered)
        # AgentRunner.chat returns an AgentResponse; surface its text.
        return getattr(response, "text_content", None) or str(response)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "prompt": self.prompt,
            "agent_factory_id": self.agent_factory_id,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            prompt=data.get("prompt", ""),
            agent_factory_id=data.get("agent_factory_id", "default"),
        )


@dataclass
class ToolNode(WorkflowNode):
    """Invoke a registered callable with arguments rendered from state."""

    tool_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_type = "tool"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        tool = executor.get_tool(self.tool_id)
        args = {
            k: _render(v, ctx.state) if isinstance(v, str) else v for k, v in self.arguments.items()
        }
        result = tool(**args)
        # Allow either sync or async tools.
        import inspect

        if inspect.isawaitable(result):
            result = await result
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "tool_id": self.tool_id,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ToolNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            tool_id=data.get("tool_id", ""),
            arguments=dict(data.get("arguments", {})),
        )


@dataclass
class BranchNode(WorkflowNode):
    """Pick the next node based on the first matching condition.

    Each entry in ``branches`` is ``(condition_expr, next_node_id)``.  The
    expression is evaluated against a restricted namespace containing the
    workflow state.  If no condition matches and ``default`` is set, it is
    used; otherwise the node's ``next`` is used.
    """

    branches: list[tuple[str, str]] = field(default_factory=list)
    default: str | None = None

    def __post_init__(self) -> None:
        self.node_type = "branch"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        for expr, target in self.branches:
            if _safe_eval(expr, ctx.state):
                # Override the next pointer for this traversal.
                ctx.state[f"__branch__{self.id}"] = target
                return target
        if self.default is not None:
            ctx.state[f"__branch__{self.id}"] = self.default
            return self.default
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "branches": [list(b) for b in self.branches],
            "default": self.default,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> BranchNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            branches=[(b[0], b[1]) for b in data.get("branches", [])],
            default=data.get("default"),
        )


@dataclass
class ParallelNode(WorkflowNode):
    """Run several branches concurrently and gather their outputs.

    Each branch is an independent sub-graph identified by its start-node id;
    execution proceeds until each branch hits a terminal (``next=None``).  The
    outputs are stored as a dict ``{branch_id: terminal_output}`` keyed by the
    start id of the branch.
    """

    branches: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = "parallel"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        import asyncio

        async def _run_branch(start_id: str) -> tuple[str, Any]:
            sub_ctx = WorkflowContext(state=dict(ctx.state))
            out = await executor._run_from(start_id, sub_ctx)
            # Merge sub-state back into parent (sub state takes precedence
            # only for keys it produced, not pre-existing ones).
            for k, v in sub_ctx.state.items():
                ctx.state.setdefault(k, v)
            ctx.history.extend(sub_ctx.history)
            return start_id, out

        results = await asyncio.gather(*(_run_branch(b) for b in self.branches))
        return {bid: out for bid, out in results}

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "branches": list(self.branches)}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ParallelNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            branches=list(data.get("branches", [])),
        )


@dataclass
class RetryNode(WorkflowNode):
    """Retry a single child node up to ``max_attempts`` times with backoff."""

    child_id: str = ""
    max_attempts: int = 3
    backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        self.node_type = "retry"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        import asyncio

        last_error: Exception | None = None
        child = executor.workflow.nodes[self.child_id]
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await child.run(ctx, executor)
            except Exception as exc:  # noqa: BLE001 — propagated after retries
                last_error = exc
                if attempt < self.max_attempts:
                    await asyncio.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "child_id": self.child_id,
            "max_attempts": self.max_attempts,
            "backoff_seconds": self.backoff_seconds,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> RetryNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            child_id=data.get("child_id", ""),
            max_attempts=int(data.get("max_attempts", 3)),
            backoff_seconds=float(data.get("backoff_seconds", 0.5)),
        )


@dataclass
class CheckpointNode(WorkflowNode):
    """Persist the current :class:`WorkflowContext` via the executor's sink.

    The executor calls ``executor.checkpoint_sink(workflow_id, ctx)`` if a
    sink is configured.  This is the seam FLOW-2 will use for durable resume.
    """

    label: str = ""

    def __post_init__(self) -> None:
        self.node_type = "checkpoint"

    async def run(self, ctx: WorkflowContext, executor: WorkflowExecutor) -> Any:
        if executor.checkpoint_sink is not None:
            executor.checkpoint_sink(executor.workflow.id, self.label, ctx)
        return {"checkpoint": self.label, "history_len": len(ctx.history)}

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "label": self.label}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CheckpointNode:
        return cls(
            id=data["id"],
            next=data.get("next"),
            label=data.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    """A serialisable directed graph of :class:`WorkflowNode` objects."""

    id: str
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    start: str = ""

    def add(self, node: WorkflowNode) -> WorkflowNode:
        if node.id in self.nodes:
            raise ValueError(f"node {node.id!r} already exists")
        self.nodes[node.id] = node
        if not self.start:
            self.start = node.id
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        wf = cls(id=data["id"], start=data.get("start", ""))
        for nid, ndata in data.get("nodes", {}).items():
            ndata = {**ndata, "id": nid}
            wf.nodes[nid] = WorkflowNode.from_dict(ndata)
        return wf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(template: str, state: dict[str, Any]) -> str:
    """Substitute ``{key}`` placeholders from state, leaving unknown keys."""
    if not isinstance(template, str) or "{" not in template:
        return template
    try:
        return template.format_map(_SafeDict(state))
    except Exception:
        return template


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "any": any,
    "all": all,
    "sum": sum,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


def _safe_eval(expr: str, state: dict[str, Any]) -> bool:
    """Evaluate ``expr`` against ``state``.  Restricted builtins only."""
    try:
        return bool(eval(expr, {"__builtins__": _SAFE_BUILTINS}, dict(state)))
    except Exception:
        return False


_NODE_REGISTRY: dict[str, type[WorkflowNode]] = {
    "agent": AgentNode,
    "tool": ToolNode,
    "branch": BranchNode,
    "parallel": ParallelNode,
    "retry": RetryNode,
    "checkpoint": CheckpointNode,
}
