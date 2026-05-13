"""Durable persistence for workflows (FLOW-2).

A :class:`WorkflowStore` writes the workflow graph and its evolving
:class:`WorkflowContext` to a directory of JSON files, one per session.
This lets a long-running workflow survive process restarts: re-instantiate
the store, call :meth:`WorkflowStore.load`, and feed the result to
:meth:`WorkflowExecutor.resume`.

Layout::

    <root>/
        <session-id>/
            workflow.json      # the DAG (immutable for the run)
            context.json       # latest WorkflowContext snapshot
            history.jsonl      # append-only audit log of every NodeResult
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from skillengine.workflow.models import NodeResult, Workflow, WorkflowContext


@dataclass
class WorkflowRecord:
    """A single durable workflow run."""

    session_id: str
    workflow: Workflow
    context: WorkflowContext
    created_at: float
    updated_at: float


class WorkflowStore:
    """File-system backed store for durable workflow runs."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, workflow: Workflow, session_id: str | None = None) -> str:
        """Register a new run.  Returns the session id."""
        sid = session_id or str(uuid.uuid4())
        path = self._session_dir(sid)
        if path.exists():
            raise FileExistsError(f"workflow session {sid!r} already exists")
        path.mkdir(parents=True)
        (path / "workflow.json").write_text(json.dumps(workflow.to_dict(), indent=2))
        ctx = WorkflowContext()
        self._write_context(sid, ctx)
        now = time.time()
        (path / "meta.json").write_text(
            json.dumps({"created_at": now, "updated_at": now}, indent=2)
        )
        return sid

    def load(self, session_id: str) -> WorkflowRecord:
        path = self._session_dir(session_id)
        if not path.exists():
            raise FileNotFoundError(f"workflow session {session_id!r} not found")
        workflow = Workflow.from_dict(json.loads((path / "workflow.json").read_text()))
        context = WorkflowContext.from_dict(json.loads((path / "context.json").read_text()))
        meta = json.loads((path / "meta.json").read_text())
        return WorkflowRecord(
            session_id=session_id,
            workflow=workflow,
            context=context,
            created_at=float(meta.get("created_at", 0.0)),
            updated_at=float(meta.get("updated_at", 0.0)),
        )

    def list_sessions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def delete(self, session_id: str) -> None:
        path = self._session_dir(session_id)
        if not path.exists():
            return
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                p.rmdir()
        path.rmdir()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_context(self, session_id: str, ctx: WorkflowContext) -> None:
        self._write_context(session_id, ctx)
        meta_path = self._session_dir(session_id) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {"created_at": time.time()}
        meta["updated_at"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2))

    def append_history(self, session_id: str, result: NodeResult) -> None:
        """Append a single :class:`NodeResult` to the audit log."""
        log = self._session_dir(session_id) / "history.jsonl"
        with log.open("a") as f:
            f.write(json.dumps(result.to_dict()))
            f.write("\n")

    def read_history(self, session_id: str) -> list[NodeResult]:
        log = self._session_dir(session_id) / "history.jsonl"
        if not log.exists():
            return []
        out: list[NodeResult] = []
        for line in log.read_text().splitlines():
            if line.strip():
                out.append(NodeResult.from_dict(json.loads(line)))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        if not session_id or "/" in session_id or session_id.startswith("."):
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.root / session_id

    def _write_context(self, session_id: str, ctx: WorkflowContext) -> None:
        path = self._session_dir(session_id) / "context.json"
        path.write_text(json.dumps(ctx.to_dict(), indent=2))
