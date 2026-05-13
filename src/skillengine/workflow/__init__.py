"""Workflow DAG abstraction.

A workflow is a serialisable directed acyclic graph of nodes.  Each node
performs a unit of work — invoking an agent, calling a tool, branching on
state, fanning out in parallel, retrying a sub-node, or checkpointing state.
The :class:`WorkflowExecutor` walks the graph and produces a final
:class:`WorkflowContext` containing the accumulated state and an audit trail
of every node execution.

Designed to support FLOW-2 (durable execution + resume) by serialising the
graph + context to JSON.
"""

from __future__ import annotations

from skillengine.workflow.executor import WorkflowError, WorkflowExecutor
from skillengine.workflow.models import (
    AgentNode,
    BranchNode,
    CheckpointNode,
    NodeResult,
    ParallelNode,
    RetryNode,
    ToolNode,
    Workflow,
    WorkflowContext,
    WorkflowNode,
)

__all__ = [
    "AgentNode",
    "BranchNode",
    "CheckpointNode",
    "NodeResult",
    "ParallelNode",
    "RetryNode",
    "ToolNode",
    "Workflow",
    "WorkflowContext",
    "WorkflowError",
    "WorkflowExecutor",
    "WorkflowNode",
]
