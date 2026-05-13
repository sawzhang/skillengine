"""Tests for FLOW-2 durable execution + resume."""

from __future__ import annotations

import json

import pytest

from skillengine.workflow import (
    BranchNode,
    NodeResult,
    ToolNode,
    Workflow,
    WorkflowContext,
    WorkflowError,
    WorkflowExecutor,
    WorkflowStore,
)


def _linear_workflow() -> Workflow:
    wf = Workflow(id="durable-wf")
    wf.add(ToolNode(id="a", tool_id="echo", arguments={"v": "alpha"}, next="b"))
    wf.add(ToolNode(id="b", tool_id="echo", arguments={"v": "beta"}, next="c"))
    wf.add(ToolNode(id="c", tool_id="echo", arguments={"v": "gamma"}))
    return wf


async def test_store_create_and_load_round_trip(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    wf = _linear_workflow()
    sid = store.create(wf)
    rec = store.load(sid)
    assert rec.session_id == sid
    assert rec.workflow.id == "durable-wf"
    assert rec.workflow.start == "a"
    assert set(rec.workflow.nodes) == {"a", "b", "c"}
    assert rec.context.current_node is None
    assert rec.context.history == []


async def test_store_rejects_duplicate_session(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    wf = _linear_workflow()
    sid = store.create(wf, session_id="fixed-id")
    assert sid == "fixed-id"
    with pytest.raises(FileExistsError):
        store.create(wf, session_id="fixed-id")


async def test_store_rejects_invalid_session_id(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    wf = _linear_workflow()
    with pytest.raises(ValueError):
        store.create(wf, session_id="../escape")
    with pytest.raises(ValueError):
        store.create(wf, session_id=".hidden")


async def test_list_and_delete(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    wf = _linear_workflow()
    a = store.create(wf, session_id="s1")
    b = store.create(wf, session_id="s2")
    assert sorted(store.list_sessions()) == [a, b]
    store.delete(a)
    assert store.list_sessions() == [b]
    # idempotent
    store.delete("nonexistent")


# ---------------------------------------------------------------------------
# Auto-persistence via executor
# ---------------------------------------------------------------------------


async def test_executor_persists_context_after_each_node(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    wf = _linear_workflow()
    sid = store.create(wf)
    executor = WorkflowExecutor(
        wf,
        tools={"echo": lambda v: v},
        store=store,
        session_id=sid,
    )
    await executor.run()

    rec = store.load(sid)
    assert rec.context.current_node is None  # workflow finished
    assert [r.node_id for r in rec.context.history] == ["a", "b", "c"]
    assert rec.context.state == {"a": "alpha", "b": "beta", "c": "gamma"}

    history = store.read_history(sid)
    assert [r.node_id for r in history] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Resume after failure
# ---------------------------------------------------------------------------


async def test_resume_continues_after_transient_failure(tmp_path) -> None:
    wf = Workflow(id="wf")
    wf.add(ToolNode(id="a", tool_id="ok", arguments={}, next="b"))
    wf.add(ToolNode(id="b", tool_id="flaky", arguments={}, next="c"))
    wf.add(ToolNode(id="c", tool_id="ok", arguments={}))

    counter = {"n": 0}

    def _flaky() -> str:
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("first attempt fails")
        return "recovered"

    tools = {"ok": lambda: "ok", "flaky": _flaky}

    store = WorkflowStore(tmp_path)
    sid = store.create(wf)
    exec1 = WorkflowExecutor(wf, tools=tools, store=store, session_id=sid)
    with pytest.raises(WorkflowError):
        await exec1.run()

    # Persistence should show that "a" completed and "b" failed.
    rec_mid = store.load(sid)
    assert rec_mid.context.current_node == "b"
    assert "a" in rec_mid.context.state
    assert "b" not in rec_mid.context.state  # failed nodes don't populate state
    # The on-disk history has both successful and failed entries.
    disk_history = store.read_history(sid)
    assert [r.node_id for r in disk_history] == ["a", "b"]
    assert disk_history[-1].error is not None

    # Resume from the failure point with a fresh context loaded from disk.
    # Trim the failed node's audit entry so re-execution records a clean one.
    rec_mid.context.history = [h for h in rec_mid.context.history if h.error is None]
    exec2 = WorkflowExecutor(wf, tools=tools, store=store, session_id=sid)
    await exec2.resume(rec_mid.context)

    rec_done = store.load(sid)
    assert rec_done.context.current_node is None
    assert rec_done.context.state["b"] == "recovered"
    assert rec_done.context.state["c"] == "ok"


# ---------------------------------------------------------------------------
# Branch + persistence interaction
# ---------------------------------------------------------------------------


async def test_branch_persisted_path(tmp_path) -> None:
    wf = Workflow(id="wf")
    wf.add(BranchNode(id="b", branches=[("x > 5", "high")], default="low"))
    wf.add(ToolNode(id="high", tool_id="t", arguments={}))
    wf.add(ToolNode(id="low", tool_id="t", arguments={}))

    store = WorkflowStore(tmp_path)
    sid = store.create(wf)
    executor = WorkflowExecutor(
        wf, tools={"t": lambda: "done"}, store=store, session_id=sid
    )
    await executor.run(initial_state={"x": 10})

    rec = store.load(sid)
    history_ids = [r.node_id for r in rec.context.history]
    assert history_ids == ["b", "high"]


# ---------------------------------------------------------------------------
# Audit log is JSONL
# ---------------------------------------------------------------------------


async def test_history_jsonl_is_one_object_per_line(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    sid = store.create(_linear_workflow())
    store.append_history(sid, NodeResult(node_id="x", node_type="tool", output="hi"))
    store.append_history(sid, NodeResult(node_id="y", node_type="tool", output=42))

    log = (tmp_path / sid / "history.jsonl").read_text().splitlines()
    assert len(log) == 2
    parsed = [json.loads(line) for line in log]
    assert parsed[0]["node_id"] == "x"
    assert parsed[1]["output"] == 42


async def test_executor_without_store_does_not_persist(tmp_path) -> None:
    """Sanity check — executor with no store should never touch disk."""
    wf = _linear_workflow()
    executor = WorkflowExecutor(wf, tools={"echo": lambda v: v})
    ctx = await executor.run()
    assert ctx.state["c"] == "gamma"
    assert list(tmp_path.iterdir()) == []  # tmp_path is unused on disk


# ---------------------------------------------------------------------------
# Context schema round-trip on disk
# ---------------------------------------------------------------------------


async def test_context_disk_format_matches_to_dict(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    sid = store.create(_linear_workflow())
    ctx = WorkflowContext(state={"x": 1}, current_node="b")
    store.save_context(sid, ctx)

    raw = json.loads((tmp_path / sid / "context.json").read_text())
    assert raw["state"] == {"x": 1}
    assert raw["current_node"] == "b"
