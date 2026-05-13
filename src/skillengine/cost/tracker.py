"""Core cost tracking primitives."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillengine.model_registry import ModelRegistry, TokenUsage

if TYPE_CHECKING:
    from skillengine.agent import AgentRunner


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    """A single billable LLM call."""

    timestamp: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    total_cost: float = 0.0
    session_id: str = ""
    skill: str = ""
    turn: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.thinking_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostEntry:
        # Tolerate extra keys.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CostSummary:
    """Aggregated cost grouped by an arbitrary key."""

    group_by: str
    rows: list[dict[str, Any]]
    total_cost: float
    total_tokens: int
    entry_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_by": self.group_by,
            "rows": self.rows,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "entry_count": self.entry_count,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


_GROUP_KEYS = {"model", "skill", "session", "session_id", "turn", "day"}


class CostTracker:
    """Records ``CostEntry`` items in memory and (optionally) to JSONL.

    Use :func:`attach_cost_tracker` to wire one to an :class:`AgentRunner`, or
    call :meth:`record` directly for offline / batch scoring.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        if self.registry.count == 0:
            # Best effort: load default catalog so cost calculations work.
            try:
                self.registry.load_defaults()
            except Exception:  # pragma: no cover - defensive
                pass
        self.log_path = Path(log_path) if log_path else None
        self.entries: list[CostEntry] = []
        if self.log_path and self.log_path.exists():
            self.entries.extend(load_entries(self.log_path))

    # ---- recording -------------------------------------------------------

    def record(
        self,
        model: str,
        usage: TokenUsage,
        *,
        session_id: str = "",
        skill: str = "",
        turn: int = 0,
        tags: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> CostEntry:
        """Record a single LLM call. Returns the entry that was stored."""
        breakdown = self.registry.calculate_cost(model, usage)
        entry = CostEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            thinking_tokens=usage.thinking_tokens,
            input_cost=breakdown.input,
            output_cost=breakdown.output,
            cache_read_cost=breakdown.cache_read,
            cache_write_cost=breakdown.cache_write,
            total_cost=breakdown.total,
            session_id=session_id,
            skill=skill,
            turn=turn,
            tags=dict(tags or {}),
        )
        self.entries.append(entry)
        if self.log_path is not None:
            _append_jsonl(self.log_path, entry)
        return entry

    # ---- aggregation -----------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(e.total_cost for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.entries)

    def summary(self, group_by: str = "model") -> CostSummary:
        """Return an aggregated summary grouped by model/skill/session/turn/day."""
        if group_by not in _GROUP_KEYS:
            raise ValueError(f"group_by must be one of {sorted(_GROUP_KEYS)}, got {group_by!r}")

        key_fn = _group_key_fn(group_by)

        buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "key": "",
                "entry_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            }
        )

        for e in self.entries:
            key = key_fn(e)
            row = buckets[key]
            row["key"] = key
            row["entry_count"] += 1
            row["input_tokens"] += e.input_tokens
            row["output_tokens"] += e.output_tokens
            row["cache_read_tokens"] += e.cache_read_tokens
            row["cache_write_tokens"] += e.cache_write_tokens
            row["thinking_tokens"] += e.thinking_tokens
            row["total_tokens"] += e.total_tokens
            row["input_cost"] += e.input_cost
            row["output_cost"] += e.output_cost
            row["total_cost"] += e.total_cost

        rows = sorted(buckets.values(), key=lambda r: -r["total_cost"])
        return CostSummary(
            group_by=group_by,
            rows=rows,
            total_cost=self.total_cost,
            total_tokens=self.total_tokens,
            entry_count=len(self.entries),
        )

    def reset(self) -> None:
        """Forget all in-memory entries. Does not delete the log file."""
        self.entries.clear()

    def to_jsonl(self, path: str | Path) -> None:
        """Write all in-memory entries to a JSONL file (overwrites)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.to_dict(), default=str) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, registry: ModelRegistry | None = None) -> CostTracker:
        """Load a tracker from a JSONL log file."""
        tracker = cls(registry=registry)
        tracker.entries.extend(load_entries(path))
        return tracker


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _group_key_fn(group_by: str) -> Callable[[CostEntry], str]:
    if group_by == "model":
        return lambda e: e.model or "<unknown>"
    if group_by == "skill":
        return lambda e: e.skill or "<none>"
    if group_by in ("session", "session_id"):
        return lambda e: e.session_id or "<none>"
    if group_by == "turn":
        return lambda e: f"turn-{e.turn}"
    if group_by == "day":
        import datetime as _dt

        return lambda e: _dt.datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
    raise ValueError(group_by)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, entry: CostEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")


def load_entries(path: str | Path) -> list[CostEntry]:
    """Load CostEntry items from a JSONL file. Skips malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[CostEntry] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                out.append(CostEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def attach_cost_tracker(
    runner: AgentRunner,
    *,
    tracker: CostTracker | None = None,
    log_path: str | Path | None = None,
) -> CostTracker:
    """Attach a :class:`CostTracker` to an :class:`AgentRunner`.

    Hooks ``TURN_END`` and records cost based on the most recent assistant
    message's ``token_usage``. The handler is non-blocking — recording errors
    are swallowed.
    """
    from skillengine.events import TURN_END

    if tracker is None:
        # Reuse the runner's registry if available for accurate pricing.
        runner_registry = getattr(runner, "model_registry", None)
        tracker = CostTracker(registry=runner_registry, log_path=log_path)

    async def _on_turn_end(payload: Any) -> None:
        try:
            messages = getattr(runner, "conversation", None) or getattr(runner, "messages", None)
            if not messages:
                return
            last = None
            for m in reversed(messages):
                if getattr(m, "role", None) == "assistant" and getattr(m, "token_usage", None):
                    last = m
                    break
            if last is None:
                return
            model_id = (
                getattr(last, "metadata", {}).get("model")
                or getattr(runner.config, "model", "")
                or "<unknown>"
            )
            current_skill = getattr(runner, "_current_skill", "") or ""
            session_id = getattr(runner, "session_id", "") or ""
            turn = getattr(payload, "turn", 0)
            tracker.record(
                model=model_id,
                usage=last.token_usage,
                session_id=str(session_id),
                skill=str(current_skill),
                turn=int(turn),
            )
        except Exception:  # pragma: no cover - defensive
            return

    runner.events.on(TURN_END, _on_turn_end, priority=50, source="cost-tracker")
    return tracker
