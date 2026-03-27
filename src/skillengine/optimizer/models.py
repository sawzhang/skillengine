"""Data models for the SkillOptimizer self-improving loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skillengine.model_registry import TokenUsage


@dataclass
class OptimizerConfig:
    """Configuration for the SkillOptimizer loop.

    Tier defaults mirror HarnessConfig conventions where applicable.
    """

    # Per-phase model overrides (None = inherit base_config model)
    scorer_model: str | None = None
    mutator_model: str | None = None

    # Stability: how many times to re-score a candidate before accepting
    stability_runs: int = 3

    # Convergence: score required to accept a mutation
    pass_threshold: float = 0.85

    # Noise floor: candidate must beat baseline by at least this margin
    improvement_margin: float = 0.02

    # Hard cap on mutation rounds
    max_rounds: int = 10

    # Max agent turns per phase (scorers and mutators are cheap — keep them short)
    scorer_max_turns: int = 5
    mutator_max_turns: int = 5

    # Changelog file written into the skill directory
    changelog_filename: str = "OPTIMIZER_CHANGELOG.md"


@dataclass
class CriterionScore:
    """Score for one checklist criterion on one test input."""

    criterion: str
    passed: bool
    score: float  # 0.0–1.0
    rationale: str = ""


@dataclass
class ScoredRun:
    """Result of running the skill against one test input and scoring the output."""

    test_input: str
    skill_output: str
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    aggregate_score: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    duration_seconds: float = 0.0


@dataclass
class MutationRecord:
    """One mutation attempt: proposed change, before/after scores, accept/reject."""

    round_number: int
    mutation_description: str
    original_content: str
    mutated_content: str
    baseline_score: float
    candidate_scores: list[float] = field(default_factory=list)
    candidate_mean: float = 0.0
    accepted: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class OptimizationReport:
    """Final report returned by SkillOptimizer.run()."""

    skill_path: Path
    initial_score: float
    final_score: float
    rounds_run: int
    mutations: list[MutationRecord] = field(default_factory=list)
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    total_duration_seconds: float = 0.0
    converged: bool = False

    @property
    def accepted_mutations(self) -> list[MutationRecord]:
        return [m for m in self.mutations if m.accepted]

    @property
    def rejected_mutations(self) -> list[MutationRecord]:
        return [m for m in self.mutations if not m.accepted]
