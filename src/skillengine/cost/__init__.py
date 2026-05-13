"""COST-1: Cost dashboard — per-turn / per-skill / per-session cost rollup.

Lightweight, opt-in cost tracking that listens for ``TURN_END`` (or accepts
manual ``record()`` calls), persists entries to JSONL, and renders a summary
table grouped by model, skill, or session.

Example::

    from skillengine import CostTracker, attach_cost_tracker

    tracker = attach_cost_tracker(agent, log_path="costs.jsonl")
    await agent.chat("hello")
    print(tracker.summary(group_by="model"))
"""

from __future__ import annotations

from skillengine.cost.tracker import (
    CostEntry,
    CostSummary,
    CostTracker,
    attach_cost_tracker,
)

__all__ = [
    "CostEntry",
    "CostSummary",
    "CostTracker",
    "attach_cost_tracker",
]
