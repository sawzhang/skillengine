"""Tests for the FLOW-1 Workflow DAG abstraction."""

from __future__ import annotations

import asyncio
import json

import pytest

from skillengine.workflow import (
    AgentNode,
    BranchNode,
    CheckpointNode,
    ParallelNode,
    RetryNode,
    ToolNode,
    Workflow,
    WorkflowContext,
    WorkflowError,
    WorkflowExecutor,
    WorkflowNode,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text_content = text


class _FakeAgent:
    """Minimal stand-in for AgentRunner — exposes only ``chat``."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    async def chat(self, prompt: str) -> _FakeResponse:
        self.calls.append(prompt)
        if not self._replies:
            return _FakeResponse("")
        return _FakeResponse(self._replies.pop(0))


# ---------------------------------------------------------------------------
# Node-level tests
# ---------------------------------------------------------------------------


async def test_tool_node_runs_sync_tool() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="t1", tool_id="add", arguments={"a": 2, "b": 3}))
    exec_ = WorkflowExecutor(wf, tools={"add": lambda a, b: a + b})
    ctx = await exec_.run()
    assert ctx.state["t1"] == 5
    assert len(ctx.history) == 1
    assert ctx.history[0].node_type == "tool"


async def test_tool_node_runs_async_tool() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="t1", tool_id="echo", arguments={"x": "hello"}))

    async def _echo(x: str) -> str:
        await asyncio.sleep(0)
        return x.upper()

    exec_ = WorkflowExecutor(wf, tools={"echo": _echo})
    ctx = await exec_.run()
    assert ctx.state["t1"] == "HELLO"


async def test_tool_node_renders_placeholders() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="t1", tool_id="greet", arguments={"who": "{name}"}))
    exec_ = WorkflowExecutor(wf, tools={"greet": lambda who: f"hi {who}"})
    ctx = await exec_.run(initial_state={"name": "alice"})
    assert ctx.state["t1"] == "hi alice"


async def test_agent_node_invokes_agent() -> None:
    wf = Workflow(id="wf")
    wf.add(AgentNode(id="a1", prompt="say {greeting}", agent_factory_id="main"))
    fake = _FakeAgent(["hello world"])
    exec_ = WorkflowExecutor(wf, agents={"main": fake})
    ctx = await exec_.run(initial_state={"greeting": "hi"})
    assert ctx.state["a1"] == "hello world"
    assert fake.calls == ["say hi"]


async def test_agent_node_missing_agent_raises() -> None:
    wf = Workflow(id="wf")
    wf.add(AgentNode(id="a1", prompt="x"))
    exec_ = WorkflowExecutor(wf)
    with pytest.raises(WorkflowError):
        await exec_.run()


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------


async def test_branch_node_picks_first_matching_branch() -> None:
    wf = Workflow(id="wf")
    wf.add(BranchNode(id="b1", branches=[("x > 5", "high"), ("x > 0", "low")], default="zero"))
    wf.add(ToolNode(id="high", tool_id="report", arguments={"tag": "HIGH"}))
    wf.add(ToolNode(id="low", tool_id="report", arguments={"tag": "LOW"}))
    wf.add(ToolNode(id="zero", tool_id="report", arguments={"tag": "ZERO"}))
    seen: list[str] = []
    exec_ = WorkflowExecutor(wf, tools={"report": lambda tag: seen.append(tag) or tag})

    await exec_.run(initial_state={"x": 10})
    assert seen == ["HIGH"]

    seen.clear()
    await exec_.run(initial_state={"x": 1})
    assert seen == ["LOW"]

    seen.clear()
    await exec_.run(initial_state={"x": 0})
    assert seen == ["ZERO"]


async def test_branch_node_default_when_no_match_and_no_default_terminates() -> None:
    wf = Workflow(id="wf")
    wf.add(BranchNode(id="b1", branches=[("x > 100", "never")]))
    wf.add(ToolNode(id="never", tool_id="boom", arguments={}))
    exec_ = WorkflowExecutor(wf, tools={"boom": lambda: pytest.fail("should not run")})
    ctx = await exec_.run(initial_state={"x": 1})
    # No match, no default → workflow terminates at the branch.
    assert ctx.history[-1].node_id == "b1"


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


async def test_parallel_node_runs_branches_concurrently() -> None:
    wf = Workflow(id="wf")
    wf.add(ParallelNode(id="p", branches=["a", "b"]))
    wf.add(ToolNode(id="a", tool_id="slow"))
    wf.add(ToolNode(id="b", tool_id="slow"))

    async def _slow() -> str:
        await asyncio.sleep(0.05)
        return "done"

    exec_ = WorkflowExecutor(wf, tools={"slow": _slow})
    import time as _time

    t0 = _time.perf_counter()
    ctx = await exec_.run()
    elapsed = _time.perf_counter() - t0
    assert elapsed < 0.15  # well under 2x serial
    assert ctx.state["p"] == {"a": "done", "b": "done"}


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


async def test_retry_node_retries_until_success() -> None:
    wf = Workflow(id="wf")
    wf.add(RetryNode(id="r", child_id="flaky", max_attempts=3, backoff_seconds=0))
    wf.add(ToolNode(id="flaky", tool_id="flaky"))

    counter = {"n": 0}

    def _flaky() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    exec_ = WorkflowExecutor(wf, tools={"flaky": _flaky})
    ctx = await exec_.run()
    assert counter["n"] == 3
    assert ctx.state["r"] == "ok"


async def test_retry_node_gives_up_after_max_attempts() -> None:
    wf = Workflow(id="wf")
    wf.add(RetryNode(id="r", child_id="bad", max_attempts=2, backoff_seconds=0))
    wf.add(ToolNode(id="bad", tool_id="bad"))

    def _bad() -> None:
        raise RuntimeError("always fails")

    exec_ = WorkflowExecutor(wf, tools={"bad": _bad})
    with pytest.raises(WorkflowError):
        await exec_.run()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


async def test_checkpoint_node_invokes_sink() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="t1", tool_id="t", arguments={}, next="cp"))
    wf.add(CheckpointNode(id="cp", label="midpoint"))
    seen: list[tuple[str, str, int]] = []

    def _sink(wf_id: str, label: str, ctx: WorkflowContext) -> None:
        seen.append((wf_id, label, len(ctx.history)))

    exec_ = WorkflowExecutor(
        wf,
        tools={"t": lambda: "x"},
        checkpoint_sink=_sink,
    )
    await exec_.run()
    assert seen == [("wf", "midpoint", 1)]


# ---------------------------------------------------------------------------
# Linear chains and traversal
# ---------------------------------------------------------------------------


async def test_linear_chain_follows_next_pointer() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="a", tool_id="get", arguments={"k": "x"}, next="b"))
    wf.add(ToolNode(id="b", tool_id="get", arguments={"k": "y"}))
    exec_ = WorkflowExecutor(wf, tools={"get": lambda k: f"value-{k}"})
    ctx = await exec_.run()
    assert [r.node_id for r in ctx.history] == ["a", "b"]
    assert ctx.state["a"] == "value-x"
    assert ctx.state["b"] == "value-y"


async def test_executor_raises_on_unknown_next() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="a", tool_id="t", next="missing"))
    exec_ = WorkflowExecutor(wf, tools={"t": lambda: 1})
    with pytest.raises(WorkflowError):
        await exec_.run()


async def test_max_steps_protects_against_infinite_loops() -> None:
    wf = Workflow(id="wf")
    # Self-cycle via a branch that always picks itself.
    wf.add(BranchNode(id="b", branches=[("True", "b")], default="b"))
    exec_ = WorkflowExecutor(wf, max_steps=5)
    with pytest.raises(WorkflowError, match="max_steps"):
        await exec_.run()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


async def test_workflow_round_trips_through_json() -> None:
    wf = Workflow(id="wf")
    wf.add(AgentNode(id="a", prompt="hi {x}", next="t"))
    wf.add(ToolNode(id="t", tool_id="echo", arguments={"k": "v"}, next="b"))
    wf.add(BranchNode(id="b", branches=[("True", "p")]))
    wf.add(ParallelNode(id="p", branches=["leaf1", "leaf2"], next="r"))
    wf.add(ToolNode(id="leaf1", tool_id="echo"))
    wf.add(ToolNode(id="leaf2", tool_id="echo"))
    wf.add(RetryNode(id="r", child_id="t", next="cp"))
    wf.add(CheckpointNode(id="cp", label="final"))

    blob = json.dumps(wf.to_dict())
    restored = Workflow.from_dict(json.loads(blob))

    assert restored.id == wf.id
    assert restored.start == wf.start
    assert set(restored.nodes) == set(wf.nodes)
    for nid in wf.nodes:
        assert type(restored.nodes[nid]) is type(wf.nodes[nid])
    assert restored.nodes["a"].prompt == "hi {x}"  # type: ignore[attr-defined]
    assert restored.nodes["t"].arguments == {"k": "v"}  # type: ignore[attr-defined]
    assert restored.nodes["r"].max_attempts == 3  # type: ignore[attr-defined]


async def test_unknown_node_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown workflow node type"):
        WorkflowNode.from_dict({"type": "bogus", "id": "x"})


async def test_context_round_trips() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="t", tool_id="echo", arguments={}))
    exec_ = WorkflowExecutor(wf, tools={"echo": lambda: "hi"})
    ctx = await exec_.run(initial_state={"k": "v"})

    blob = json.dumps(ctx.to_dict())
    restored = WorkflowContext.from_dict(json.loads(blob))
    assert restored.state == ctx.state
    assert len(restored.history) == len(ctx.history)
    assert restored.history[0].node_id == "t"


# ---------------------------------------------------------------------------
# Resume (FLOW-2 seam)
# ---------------------------------------------------------------------------


async def test_resume_continues_from_current_node() -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="a", tool_id="get", arguments={}, next="b"))
    wf.add(ToolNode(id="b", tool_id="get", arguments={}))
    exec_ = WorkflowExecutor(wf, tools={"get": lambda: "x"})

    ctx = WorkflowContext(current_node="b")
    await exec_.resume(ctx)
    assert [r.node_id for r in ctx.history] == ["b"]
