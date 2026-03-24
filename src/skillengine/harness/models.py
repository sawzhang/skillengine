"""Data models for the harness system."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillengine.model_registry import TokenUsage


@dataclass
class HarnessConfig:
    """Configuration for the harness orchestrator.

    Tier selection:
    - enable_planner=False, enable_sprints=False → Tier 1 (Generator + Evaluator)
    - enable_planner=True,  enable_sprints=False → Tier 2 (+ Planner)
    - enable_planner=True,  enable_sprints=True  → Tier 3 (+ Sprint decomposition)
    """

    # Complexity tier
    enable_planner: bool = False
    enable_sprints: bool = False

    # Per-phase model overrides (None = use base config model)
    planner_model: str | None = None
    generator_model: str | None = None
    evaluator_model: str | None = None

    # Iteration limits
    max_refinement_rounds: int = 3
    max_sprints: int = 10

    # Per-phase turn limits
    planner_max_turns: int = 10
    generator_max_turns: int = 50
    evaluator_max_turns: int = 10

    # Evaluation
    pass_threshold: float = 0.8

    # Per-phase tool restrictions (None = all tools)
    generator_tools: list[str] | None = None
    evaluator_tools: list[str] | None = None


@dataclass
class SprintContract:
    """Definition of work for one sprint."""

    sprint_number: int
    title: str
    description: str
    acceptance_criteria: list[str]
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"


@dataclass
class EvaluationResult:
    """Structured output from the Evaluator agent."""

    passed: bool
    score: float
    criteria_results: dict[str, bool]
    feedback: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result from a single harness phase execution."""

    phase: str
    sprint_number: int = 0
    refinement_round: int = 0
    output: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    duration_seconds: float = 0.0
    evaluation: EvaluationResult | None = None


@dataclass
class SprintResult:
    """Aggregate result for one sprint's Generator↔Evaluator loop."""

    contract: SprintContract
    phases: list[PhaseResult] = field(default_factory=list)
    final_passed: bool = False
    total_refinement_rounds: int = 0

    @property
    def total_usage(self) -> TokenUsage:
        usage = TokenUsage()
        for p in self.phases:
            usage += p.token_usage
        return usage


@dataclass
class HarnessReport:
    """Final report for the entire harness run."""

    sprints: list[SprintResult] = field(default_factory=list)
    planner_result: PhaseResult | None = None
    total_duration_seconds: float = 0.0

    @property
    def total_usage(self) -> TokenUsage:
        usage = TokenUsage()
        if self.planner_result:
            usage += self.planner_result.token_usage
        for sprint in self.sprints:
            usage += sprint.total_usage
        return usage

    def cost_breakdown_by_phase(self) -> dict[str, TokenUsage]:
        """Return token usage keyed by phase label.

        Keys like: "planner", "generator_S1_R0", "evaluator_S1_R0".
        """
        breakdown: dict[str, TokenUsage] = {}
        if self.planner_result:
            breakdown["planner"] = self.planner_result.token_usage
        for sprint in self.sprints:
            for phase in sprint.phases:
                key = f"{phase.phase}_S{phase.sprint_number}_R{phase.refinement_round}"
                breakdown[key] = phase.token_usage
        return breakdown
