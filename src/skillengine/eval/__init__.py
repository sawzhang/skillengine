"""SkillEngine evaluation harness (EVAL-1).

Lightweight, dependency-free regression / benchmarking harness for agents and
skills. Provides:

* :class:`EvalCase` — one input + one expected outcome.
* :class:`EvalDataset` — a collection of cases with JSON/JSONL I/O.
* :class:`Scorer` (abstract) + built-in scorers:

  - :class:`ExactMatchScorer`
  - :class:`ContainsScorer`
  - :class:`RegexScorer`
  - :class:`StructuredMatchScorer`
  - :class:`LLMJudgeScorer`

* :class:`EvalRunner` — runs a target callable over a dataset, applies
  scorers, and produces an :class:`EvalReport`.
* Built-in suite ``skill-dsl`` — 30+ regression cases for the skill DSL
  loader/filter pipeline.

Hooked into the CLI as ``skills eval``.
"""

from .dataset import EvalCase, EvalDataset
from .runner import EvalCaseResult, EvalReport, EvalRunner
from .scorers import (
    ContainsScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    RegexScorer,
    Scorer,
    ScorerResult,
    StructuredMatchScorer,
)
from .suites import builtin_suite, list_builtin_suites

__all__ = [
    "EvalCase",
    "EvalDataset",
    "EvalCaseResult",
    "EvalReport",
    "EvalRunner",
    "Scorer",
    "ScorerResult",
    "ExactMatchScorer",
    "ContainsScorer",
    "RegexScorer",
    "StructuredMatchScorer",
    "LLMJudgeScorer",
    "builtin_suite",
    "list_builtin_suites",
]
