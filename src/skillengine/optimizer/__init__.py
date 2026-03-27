"""SkillOptimizer — Autoresearch self-improving skill loop.

Implements the Autoresearch hill-climbing pattern for SkillEngine:
mutate skill prompt → score × stability_runs → keep if better, revert if worse
→ repeat until pass_threshold or max_rounds.

Usage::

    from skillengine.optimizer import SkillOptimizer, OptimizerConfig

    optimizer = SkillOptimizer(
        config=OptimizerConfig(max_rounds=5, pass_threshold=0.85),
        base_config=AgentConfig(model="claude-sonnet-4-20250514"),
    )
    report = await optimizer.run(
        skill_path=Path("skills/my-skill/SKILL.md"),
        checklist=[
            "Output is valid JSON",
            "Response addresses the user question",
        ],
        test_inputs=["Summarize this", "List open PRs"],
    )
"""

from skillengine.optimizer.models import (
    CriterionScore,
    MutationRecord,
    OptimizationReport,
    OptimizerConfig,
    ScoredRun,
)
from skillengine.optimizer.runner import SkillOptimizer

__all__ = [
    "SkillOptimizer",
    "OptimizerConfig",
    "OptimizationReport",
    "MutationRecord",
    "ScoredRun",
    "CriterionScore",
]
