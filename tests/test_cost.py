"""Tests for COST-1: cost tracking and dashboard."""

from __future__ import annotations

import json

import pytest

from skillengine import (
    CostEntry,
    CostTracker,
)
from skillengine.model_registry import ModelCost, ModelDefinition, ModelRegistry, TokenUsage


def _make_registry() -> ModelRegistry:
    r = ModelRegistry()
    r.register(
        ModelDefinition(
            id="test-cheap",
            provider="test",
            cost=ModelCost(input=1.0, output=2.0),
        )
    )
    r.register(
        ModelDefinition(
            id="test-pricey",
            provider="test",
            cost=ModelCost(input=10.0, output=30.0),
        )
    )
    return r


# ---------------------------------------------------------------------------
# CostTracker.record + summary
# ---------------------------------------------------------------------------


def test_record_calculates_cost() -> None:
    t = CostTracker(registry=_make_registry())
    entry = t.record(
        "test-cheap",
        TokenUsage(input_tokens=1_000_000, output_tokens=500_000),
    )
    # 1M @ $1 + 0.5M @ $2 = 1 + 1 = $2
    assert entry.total_cost == pytest.approx(2.0)
    assert entry.input_cost == pytest.approx(1.0)
    assert entry.output_cost == pytest.approx(1.0)
    assert t.total_cost == pytest.approx(2.0)
    assert t.total_tokens == 1_500_000


def test_record_unknown_model_zero_cost() -> None:
    t = CostTracker(registry=_make_registry())
    e = t.record("nonexistent", TokenUsage(input_tokens=1000, output_tokens=500))
    assert e.total_cost == 0.0
    assert e.input_tokens == 1000


def test_summary_groups_by_model() -> None:
    t = CostTracker(registry=_make_registry())
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000))
    t.record("test-cheap", TokenUsage(output_tokens=1_000_000))
    t.record("test-pricey", TokenUsage(input_tokens=1_000_000))

    s = t.summary(group_by="model")
    assert s.group_by == "model"
    assert s.entry_count == 3
    rows = {r["key"]: r for r in s.rows}
    assert "test-cheap" in rows
    assert rows["test-cheap"]["entry_count"] == 2
    assert rows["test-cheap"]["total_cost"] == pytest.approx(3.0)  # 1 + 2
    assert rows["test-pricey"]["total_cost"] == pytest.approx(10.0)
    # Rows sorted descending by total_cost
    assert s.rows[0]["key"] == "test-pricey"


def test_summary_group_by_skill_and_session() -> None:
    t = CostTracker(registry=_make_registry())
    t.record("test-cheap", TokenUsage(input_tokens=2_000_000), skill="github", session_id="s1")
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000), skill="pdf", session_id="s1")
    t.record("test-cheap", TokenUsage(input_tokens=500_000), skill="github", session_id="s2")

    by_skill = t.summary(group_by="skill")
    skills = {r["key"]: r["total_cost"] for r in by_skill.rows}
    assert skills["github"] == pytest.approx(2.5)
    assert skills["pdf"] == pytest.approx(1.0)

    by_session = t.summary(group_by="session")
    sessions = {r["key"]: r["entry_count"] for r in by_session.rows}
    assert sessions["s1"] == 2
    assert sessions["s2"] == 1


def test_summary_group_by_day() -> None:
    import datetime as dt

    t = CostTracker(registry=_make_registry())
    ts1 = dt.datetime(2026, 1, 1, 10, 0).timestamp()
    ts2 = dt.datetime(2026, 1, 2, 10, 0).timestamp()
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000), timestamp=ts1)
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000), timestamp=ts2)
    s = t.summary(group_by="day")
    days = {r["key"] for r in s.rows}
    assert "2026-01-01" in days
    assert "2026-01-02" in days


def test_summary_invalid_group_by_raises() -> None:
    t = CostTracker()
    with pytest.raises(ValueError):
        t.summary(group_by="bogus")


def test_reset_clears_entries() -> None:
    t = CostTracker(registry=_make_registry())
    t.record("test-cheap", TokenUsage(input_tokens=100))
    assert len(t.entries) == 1
    t.reset()
    assert t.entries == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_log_path_appends_jsonl(tmp_path) -> None:
    log = tmp_path / "costs.jsonl"
    t = CostTracker(registry=_make_registry(), log_path=log)
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000), skill="gh")
    t.record("test-cheap", TokenUsage(output_tokens=1_000_000), skill="gh")
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["model"] == "test-cheap"
    assert first["skill"] == "gh"


def test_to_jsonl_roundtrip(tmp_path) -> None:
    t = CostTracker(registry=_make_registry())
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000))
    t.record("test-pricey", TokenUsage(output_tokens=500_000))
    p = tmp_path / "out.jsonl"
    t.to_jsonl(p)
    loaded = CostTracker.from_jsonl(p)
    assert len(loaded.entries) == 2
    assert loaded.entries[0].model == "test-cheap"
    assert loaded.total_cost == pytest.approx(t.total_cost)


def test_constructor_loads_existing_log(tmp_path) -> None:
    log = tmp_path / "costs.jsonl"
    t = CostTracker(registry=_make_registry(), log_path=log)
    t.record("test-cheap", TokenUsage(input_tokens=1_000_000))

    # Re-open: should read existing entries.
    t2 = CostTracker(registry=_make_registry(), log_path=log)
    assert len(t2.entries) == 1
    # Recording in t2 appends another line.
    t2.record("test-cheap", TokenUsage(output_tokens=500_000))
    assert len(log.read_text().splitlines()) == 2


def test_from_jsonl_tolerates_malformed_lines(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"timestamp": 1, "model": "m", "input_tokens": 10, "total_cost": 0.1}\n'
        "not json\n"
        '{"timestamp": 2, "model": "m", "output_tokens": 20, "total_cost": 0.2}\n'
    )
    t = CostTracker.from_jsonl(p)
    assert len(t.entries) == 2
    assert t.entries[1].output_tokens == 20


def test_cost_entry_dict_roundtrip() -> None:
    e = CostEntry(
        timestamp=123.0,
        model="m",
        input_tokens=10,
        output_tokens=20,
        total_cost=0.5,
        skill="x",
    )
    d = e.to_dict()
    e2 = CostEntry.from_dict(d)
    assert e2.model == "m"
    assert e2.total_cost == 0.5


def test_cost_entry_from_dict_tolerates_extra_keys() -> None:
    e = CostEntry.from_dict({"timestamp": 1.0, "model": "m", "spurious": "ignored"})
    assert e.model == "m"


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_cost_tracker_records_on_turn_end() -> None:
    from skillengine import attach_cost_tracker
    from skillengine.agent import AgentMessage
    from skillengine.events import TURN_END, EventBus, TurnEndEvent

    # Minimal stand-in for AgentRunner — events bus + conversation + config.
    class _FakeRunner:
        def __init__(self) -> None:
            self.events = EventBus()
            self.conversation: list[AgentMessage] = []

            class _Cfg:
                model = "test-cheap"

            self.config = _Cfg()
            self.session_id = "sess-1"
            self._current_skill = "demo"
            self.model_registry = _make_registry()

    runner = _FakeRunner()
    tracker = attach_cost_tracker(runner)
    assert isinstance(tracker, CostTracker)

    # Simulate an assistant message arriving and a TURN_END event.
    runner.conversation.append(
        AgentMessage(
            role="assistant",
            content="hi",
            token_usage=TokenUsage(input_tokens=1_000_000, output_tokens=500_000),
        )
    )
    await runner.events.emit(
        TURN_END,
        TurnEndEvent(turn=1, has_tool_calls=False, content="hi", tool_call_count=0),
    )

    assert len(tracker.entries) == 1
    e = tracker.entries[0]
    assert e.model == "test-cheap"
    assert e.skill == "demo"
    assert e.session_id == "sess-1"
    assert e.turn == 1
    assert e.total_cost == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_attach_cost_tracker_handles_missing_usage() -> None:
    from skillengine import attach_cost_tracker
    from skillengine.events import TURN_END, EventBus, TurnEndEvent

    class _FakeRunner:
        def __init__(self) -> None:
            self.events = EventBus()
            self.conversation = []

            class _Cfg:
                model = "test-cheap"

            self.config = _Cfg()
            self.model_registry = _make_registry()

    runner = _FakeRunner()
    tracker = attach_cost_tracker(runner)
    # No assistant messages with token_usage — should not crash.
    await runner.events.emit(
        TURN_END, TurnEndEvent(turn=1, has_tool_calls=False, content="", tool_call_count=0)
    )
    assert tracker.entries == []
