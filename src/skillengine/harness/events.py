"""Harness-specific event types for the EventBus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillengine.harness.models import (
        EvaluationResult,
        HarnessConfig,
        HarnessReport,
        PhaseResult,
        SprintContract,
        SprintResult,
    )

# Event name constants
HARNESS_START = "harness_start"
HARNESS_END = "harness_end"
SPRINT_START = "sprint_start"
SPRINT_END = "sprint_end"
PHASE_START = "phase_start"
PHASE_END = "phase_end"
EVALUATION_COMPLETE = "evaluation_complete"


@dataclass
class HarnessStartEvent:
    config: HarnessConfig
    user_input: str


@dataclass
class HarnessEndEvent:
    report: HarnessReport


@dataclass
class SprintStartEvent:
    sprint_number: int
    contract: SprintContract


@dataclass
class SprintEndEvent:
    sprint_number: int
    result: SprintResult


@dataclass
class PhaseStartEvent:
    phase: str
    sprint_number: int
    refinement_round: int


@dataclass
class PhaseEndEvent:
    phase: str
    result: PhaseResult


@dataclass
class EvaluationCompleteEvent:
    sprint_number: int
    refinement_round: int
    result: EvaluationResult
