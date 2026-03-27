"""Tests for the SkillOptimizer self-improving loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
from skillengine.model_registry import TokenUsage
from skillengine.optimizer.changelog import ChangelogWriter
from skillengine.optimizer.models import (
    CriterionScore,
    MutationRecord,
    OptimizationReport,
    OptimizerConfig,
    ScoredRun,
)
from skillengine.optimizer.prompts import format_mutator_prompt, format_scorer_prompt
from skillengine.optimizer.runner import SkillOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKLIST = ["Output is clear", "Response is concise", "No hallucinations"]
TEST_INPUTS = ["Hello, help me with X", "Summarize this document"]

SKILL_CONTENT = """\
---
name: test-skill
description: A test skill
---

You are a helpful assistant. Answer the user's question clearly and concisely.
"""


def _make_agent_message(content: str) -> AgentMessage:
    return AgentMessage(role="assistant", content=content)


def _make_scorer_json(
    criteria: list[str] | None = None,
    scores: list[float] | None = None,
    aggregate: float | None = None,
) -> str:
    criteria = criteria or CHECKLIST
    scores = scores or [0.8] * len(criteria)
    agg = aggregate if aggregate is not None else sum(scores) / len(scores)
    return json.dumps(
        {
            "criterion_scores": [
                {
                    "criterion": c,
                    "passed": s >= 0.7,
                    "score": s,
                    "rationale": "test rationale",
                }
                for c, s in zip(criteria, scores)
            ],
            "aggregate_score": agg,
            "overall_feedback": "test feedback",
        }
    )


def _make_noop_mutator() -> str:
    """Mutator output that returns identical content (triggers no-op guard)."""
    return f"```skill\n{SKILL_CONTENT}\n```\n\n```mutation_description\nno change needed\n```"


def _make_mutator_output(
    new_body: str = "You are an improved assistant.",
    description: str = "Added clarity to instructions.",
) -> str:
    new_content = SKILL_CONTENT.replace(
        "You are a helpful assistant. Answer the user's question clearly and concisely.",
        new_body,
    )
    return f"```skill\n{new_content}\n```\n\n```mutation_description\n{description}\n```"


def _mock_create_agent(chat_responses: list[AgentMessage]):
    """Factory that returns mock agents with queued responses.

    Mirrors the pattern from test_harness._mock_create_agent.
    """
    call_index = [0]

    def factory(
        self,
        system_prompt,
        model_override=None,
        max_turns=None,
        enable_tools=None,
    ):
        agent = MagicMock(spec=AgentRunner)
        agent.cumulative_usage = TokenUsage()

        idx = call_index[0]
        if idx < len(chat_responses):
            agent.chat = AsyncMock(return_value=chat_responses[idx])
            call_index[0] += 1
        else:
            agent.chat = AsyncMock(return_value=_make_agent_message(""))

        return agent

    return factory


@pytest.fixture
def skill_file(tmp_path: Path) -> Path:
    """Write a minimal SKILL.md to tmp_path and return its Path."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    p = skill_dir / "SKILL.md"
    p.write_text(SKILL_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        stability_runs=1,
        max_rounds=3,
        improvement_margin=0.02,
        pass_threshold=0.85,
    )


@pytest.fixture
def base_config() -> AgentConfig:
    return AgentConfig(model="test-model", enable_tools=False)


# ---------------------------------------------------------------------------
# TestOptimizerModels
# ---------------------------------------------------------------------------


class TestOptimizerModels:
    def test_accepted_mutations_property(self):
        report = OptimizationReport(
            skill_path=Path("x/SKILL.md"),
            initial_score=0.5,
            final_score=0.8,
            rounds_run=3,
            mutations=[
                MutationRecord(1, "mut1", "", "", 0.5, accepted=True, candidate_mean=0.7),
                MutationRecord(2, "mut2", "", "", 0.7, accepted=False, candidate_mean=0.65),
                MutationRecord(3, "mut3", "", "", 0.7, accepted=True, candidate_mean=0.8),
            ],
        )
        assert len(report.accepted_mutations) == 2
        assert len(report.rejected_mutations) == 1
        assert report.accepted_mutations[0].round_number == 1
        assert report.rejected_mutations[0].round_number == 2

    def test_scored_run_default_aggregate_zero(self):
        run = ScoredRun(test_input="x", skill_output="y")
        assert run.aggregate_score == 0.0
        assert run.criterion_scores == []

    def test_optimizer_config_defaults(self):
        cfg = OptimizerConfig()
        assert cfg.stability_runs == 3
        assert cfg.pass_threshold == 0.85
        assert cfg.improvement_margin == 0.02
        assert cfg.max_rounds == 10


# ---------------------------------------------------------------------------
# TestOptimizerPrompts
# ---------------------------------------------------------------------------


class TestOptimizerPrompts:
    def test_scorer_prompt_contains_test_input(self):
        prompt = format_scorer_prompt("my test input", "skill output", CHECKLIST)
        assert "my test input" in prompt

    def test_scorer_prompt_contains_skill_output(self):
        prompt = format_scorer_prompt("input", "my skill output", CHECKLIST)
        assert "my skill output" in prompt

    def test_scorer_prompt_contains_each_criterion(self):
        prompt = format_scorer_prompt("input", "output", CHECKLIST)
        for criterion in CHECKLIST:
            assert criterion in prompt

    def test_scorer_prompt_requires_json(self):
        prompt = format_scorer_prompt("input", "output", CHECKLIST)
        assert "JSON" in prompt

    def test_mutator_prompt_contains_baseline_score(self):
        prompt = format_mutator_prompt(SKILL_CONTENT, CHECKLIST, 0.62, [], [])
        assert "0.62" in prompt

    def test_mutator_prompt_contains_skill_content(self):
        prompt = format_mutator_prompt(SKILL_CONTENT, CHECKLIST, 0.5, [], [])
        assert "test-skill" in prompt

    def test_mutator_prompt_contains_weak_criteria(self):
        weak = [("Output is clear", 0.3), ("No hallucinations", 0.4)]
        prompt = format_mutator_prompt(SKILL_CONTENT, CHECKLIST, 0.5, weak, [])
        assert "Output is clear" in prompt
        assert "0.30" in prompt

    def test_mutator_prompt_contains_prior_mutations(self):
        prior = ["Added example section.", "Removed ambiguous phrasing."]
        prompt = format_mutator_prompt(SKILL_CONTENT, CHECKLIST, 0.5, [], prior)
        assert "Added example section." in prompt
        assert "Removed ambiguous phrasing." in prompt

    def test_mutator_prompt_no_prior_shows_none_label(self):
        prompt = format_mutator_prompt(SKILL_CONTENT, CHECKLIST, 0.5, [], [])
        assert "none" in prompt.lower()


# ---------------------------------------------------------------------------
# TestParsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parse_scored_run_clean_json(self):
        raw = _make_scorer_json(scores=[0.9, 0.8, 0.7])
        result = SkillOptimizer._parse_scored_run(raw, "input", "output")
        assert len(result.criterion_scores) == 3
        assert abs(result.aggregate_score - 0.8) < 0.01

    def test_parse_scored_run_with_code_fence(self):
        raw = "Here is my scoring:\n```json\n" + _make_scorer_json() + "\n```"
        result = SkillOptimizer._parse_scored_run(raw, "input", "output")
        assert len(result.criterion_scores) == len(CHECKLIST)

    def test_parse_scored_run_garbage_returns_zeros(self):
        result = SkillOptimizer._parse_scored_run("not json at all", "input", "output")
        assert result.aggregate_score == 0.0
        assert result.criterion_scores == []

    def test_parse_scored_run_passed_field(self):
        raw = _make_scorer_json(scores=[0.9, 0.6, 0.4])
        result = SkillOptimizer._parse_scored_run(raw, "input", "output")
        assert result.criterion_scores[0].passed is True  # 0.9 >= 0.7
        assert result.criterion_scores[1].passed is False  # 0.6 < 0.7
        assert result.criterion_scores[2].passed is False  # 0.4 < 0.7

    def test_parse_mutator_output_extracts_skill_block(self):
        new_body = "You are an enhanced assistant."
        raw = _make_mutator_output(new_body=new_body)
        skill_content, _ = SkillOptimizer._parse_mutator_output(raw, SKILL_CONTENT)
        assert new_body in skill_content

    def test_parse_mutator_output_extracts_description(self):
        raw = _make_mutator_output(description="Added a worked example.")
        _, description = SkillOptimizer._parse_mutator_output(raw, SKILL_CONTENT)
        assert description == "Added a worked example."

    def test_parse_mutator_output_missing_blocks_returns_original(self):
        skill_content, desc = SkillOptimizer._parse_mutator_output("No blocks here.", SKILL_CONTENT)
        assert skill_content == SKILL_CONTENT
        assert "no parseable" in desc.lower()

    def test_extract_weak_criteria_sorted_ascending(self):
        runs = [
            ScoredRun(
                test_input="x",
                skill_output="y",
                criterion_scores=[
                    CriterionScore("A", True, 0.9),
                    CriterionScore("B", False, 0.3),
                    CriterionScore("C", False, 0.5),
                ],
            )
        ]
        weak = SkillOptimizer._extract_weak_criteria(runs, top_n=2)
        assert weak[0][0] == "B"  # lowest score first
        assert weak[1][0] == "C"
        assert len(weak) == 2

    def test_extract_weak_criteria_top_n_limit(self):
        runs = [
            ScoredRun(
                test_input="x",
                skill_output="y",
                criterion_scores=[CriterionScore(f"C{i}", False, float(i) / 10) for i in range(10)],
            )
        ]
        weak = SkillOptimizer._extract_weak_criteria(runs, top_n=3)
        assert len(weak) == 3

    def test_extract_weak_criteria_empty_runs(self):
        assert SkillOptimizer._extract_weak_criteria([]) == []


# ---------------------------------------------------------------------------
# TestSkillOptimizerRun (integration, all mocked)
# ---------------------------------------------------------------------------


class TestSkillOptimizerRun:
    async def test_converges_immediately_if_above_threshold(
        self, skill_file, optimizer_config, base_config
    ):
        """Baseline already >= pass_threshold: no mutation rounds run."""
        # Interleaved: skill_A → scorer_A → skill_B → scorer_B
        responses = [
            _make_agent_message("skill output A"),
            _make_agent_message(_make_scorer_json(scores=[0.9, 0.9, 0.9])),
            _make_agent_message("skill output B"),
            _make_agent_message(_make_scorer_json(scores=[0.9, 0.9, 0.9])),
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        assert report.converged is True
        assert report.rounds_run == 0
        assert report.mutations == []
        assert report.initial_score >= optimizer_config.pass_threshold

    async def test_mutation_accepted_when_score_improves(
        self, skill_file, optimizer_config, base_config
    ):
        """Round 1: baseline=0.5 → candidate=0.75 (accepted)."""
        low_scorer = _make_scorer_json(scores=[0.5, 0.5, 0.5])  # baseline
        high_scorer = _make_scorer_json(scores=[0.75, 0.75, 0.75])  # candidate
        mutator_resp = _make_agent_message(_make_mutator_output())

        responses = [
            # Baseline: interleaved skill/score per input
            _make_agent_message("skill out A"),
            _make_agent_message(low_scorer),
            _make_agent_message("skill out B"),
            _make_agent_message(low_scorer),
            # Mutator
            mutator_resp,
            # Stability scoring (stability_runs=1): interleaved
            _make_agent_message("skill out A2"),
            _make_agent_message(high_scorer),
            _make_agent_message("skill out B2"),
            _make_agent_message(high_scorer),
            # Refresh scored_runs after accept: interleaved
            _make_agent_message("skill out A3"),
            _make_agent_message(high_scorer),
            _make_agent_message("skill out B3"),
            _make_agent_message(high_scorer),
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        original_content = skill_file.read_text()
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        assert len(report.accepted_mutations) == 1
        assert report.mutations[0].accepted is True
        # Skill file should have been updated
        assert skill_file.read_text() != original_content

    async def test_mutation_rejected_when_score_does_not_improve(
        self, skill_file, optimizer_config, base_config
    ):
        """Round 1: baseline=0.6, candidate=0.55 (rejected, reverted)."""
        medium_scorer = _make_scorer_json(scores=[0.6, 0.6, 0.6])
        low_scorer = _make_scorer_json(scores=[0.55, 0.55, 0.55])
        mutator_resp = _make_agent_message(_make_mutator_output(new_body="Worse instructions."))

        responses = [
            # Baseline: interleaved
            _make_agent_message("out A"),
            _make_agent_message(medium_scorer),
            _make_agent_message("out B"),
            _make_agent_message(medium_scorer),
            # Mutator
            mutator_resp,
            # Stability (regressed): interleaved
            _make_agent_message("out A2"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B2"),
            _make_agent_message(low_scorer),
            # 2nd and 3rd rounds: no-op mutators to exhaust max_rounds
            _make_agent_message(_make_noop_mutator()),
            _make_agent_message(_make_noop_mutator()),
        ]
        original_content = skill_file.read_text()
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        assert len(report.rejected_mutations) >= 1
        assert report.mutations[0].accepted is False
        # File should be reverted to original
        assert skill_file.read_text() == original_content

    async def test_stops_at_max_rounds(self, skill_file, base_config):
        """Stops after max_rounds even if never converged."""
        config = OptimizerConfig(max_rounds=2, stability_runs=1, improvement_margin=0.02)
        low_scorer = _make_scorer_json(scores=[0.4, 0.4, 0.4])
        mutator_resp = _make_mutator_output()

        # Each round: baseline(2+2) + mutator(1) + stability(2+2) + refresh after accept(2+2)
        # But we want 2 rejected rounds so no refresh, each round: 4 + 1 + 4 = 9 calls
        # Keep it simple: provide enough responses
        responses = (
            [
                _make_agent_message("out A"),
                _make_agent_message(low_scorer),
                _make_agent_message("out B"),
                _make_agent_message(low_scorer),
            ]  # baseline (interleaved)
            + [_make_agent_message(mutator_resp)]  # round 1 mutator
            + [
                _make_agent_message("out A"),
                _make_agent_message(low_scorer),
                _make_agent_message("out B"),
                _make_agent_message(low_scorer),
            ]  # r1 stability (interleaved)
            + [_make_agent_message(mutator_resp)]  # round 2 mutator
            + [
                _make_agent_message("out A"),
                _make_agent_message(low_scorer),
                _make_agent_message("out B"),
                _make_agent_message(low_scorer),
            ]  # r2 stability (interleaved)
        )
        optimizer = SkillOptimizer(config=config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        assert report.rounds_run <= 2
        assert report.converged is False

    async def test_converges_mid_run(self, skill_file, optimizer_config, base_config):
        """Score crosses pass_threshold during round 1 → converged=True."""
        low_scorer = _make_scorer_json(scores=[0.5, 0.5, 0.5])
        passing_scorer = _make_scorer_json(scores=[0.9, 0.9, 0.9])

        responses = [
            # Baseline (0.5): interleaved
            _make_agent_message("out A"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B"),
            _make_agent_message(low_scorer),
            # Mutator
            _make_agent_message(_make_mutator_output()),
            # Stability (0.9 > threshold=0.85): interleaved
            _make_agent_message("out A2"),
            _make_agent_message(passing_scorer),
            _make_agent_message("out B2"),
            _make_agent_message(passing_scorer),
            # Refresh after accept: interleaved
            _make_agent_message("out A3"),
            _make_agent_message(passing_scorer),
            _make_agent_message("out B3"),
            _make_agent_message(passing_scorer),
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        assert report.converged is True
        assert report.rounds_run == 1
        assert report.final_score >= optimizer_config.pass_threshold

    async def test_no_op_mutation_skipped(self, skill_file, optimizer_config, base_config):
        """Mutator returns identical content → round skipped, no stability scoring."""
        low_scorer = _make_scorer_json(scores=[0.4, 0.4, 0.4])
        # No-op: same content as SKILL_CONTENT
        noop_mutator = _make_agent_message(
            f"```skill\n{SKILL_CONTENT}\n```\n\n```mutation_description\nno change needed\n```"
        )

        responses = [
            # Baseline: interleaved
            _make_agent_message("out A"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B"),
            _make_agent_message(low_scorer),
            # No-op mutator (round 1, 2, 3)
            noop_mutator,
            noop_mutator,
            noop_mutator,
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            report = await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        # No mutations recorded (all skipped as no-ops)
        assert len(report.mutations) == 0

    async def test_changelog_file_written(self, skill_file, optimizer_config, base_config):
        """OPTIMIZER_CHANGELOG.md is created in the skill directory."""
        low_scorer = _make_scorer_json(scores=[0.5, 0.5, 0.5])
        passing_scorer = _make_scorer_json(scores=[0.9, 0.9, 0.9])

        responses = [
            # Baseline: interleaved
            _make_agent_message("out A"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B"),
            _make_agent_message(low_scorer),
            _make_agent_message(_make_mutator_output()),
            # Stability: interleaved
            _make_agent_message("out A2"),
            _make_agent_message(passing_scorer),
            _make_agent_message("out B2"),
            _make_agent_message(passing_scorer),
            # Refresh: interleaved
            _make_agent_message("out A3"),
            _make_agent_message(passing_scorer),
            _make_agent_message("out B3"),
            _make_agent_message(passing_scorer),
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        changelog_path = skill_file.parent / "OPTIMIZER_CHANGELOG.md"
        assert changelog_path.exists()
        content = changelog_path.read_text()
        assert "Optimization Run" in content
        assert "test-skill" in content

    async def test_skill_file_atomically_written(self, skill_file, optimizer_config, base_config):
        """No SKILL.md.tmp remains after a completed run."""
        low_scorer = _make_scorer_json(scores=[0.5, 0.5, 0.5])

        responses = [
            # Baseline: interleaved
            _make_agent_message("out A"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B"),
            _make_agent_message(low_scorer),
            _make_agent_message(_make_mutator_output()),
            # Stability: interleaved
            _make_agent_message("out A2"),
            _make_agent_message(low_scorer),
            _make_agent_message("out B2"),
            _make_agent_message(low_scorer),
        ]
        optimizer = SkillOptimizer(config=optimizer_config, base_config=base_config)
        with patch.object(SkillOptimizer, "_create_agent", _mock_create_agent(responses)):
            await optimizer.run(skill_file, CHECKLIST, TEST_INPUTS)

        tmp_path = skill_file.parent / "SKILL.md.tmp"
        assert not tmp_path.exists()


# ---------------------------------------------------------------------------
# TestChangelogWriter
# ---------------------------------------------------------------------------


class TestChangelogWriter:
    def test_write_header_creates_file(self, tmp_path):
        writer = ChangelogWriter(tmp_path)
        writer.write_header("my-skill", CHECKLIST, 3, "2026-03-27T12:00:00Z")
        writer.write_footer(OptimizationReport(Path("x"), 0.5, 0.5, 0, converged=False))
        assert (tmp_path / "OPTIMIZER_CHANGELOG.md").exists()

    def test_append_round_accepted(self, tmp_path):
        writer = ChangelogWriter(tmp_path)
        writer.write_header("s", [], 1, "2026-01-01T00:00:00Z")
        writer.append_round(
            MutationRecord(1, "Added example", "", "", 0.5, candidate_mean=0.7, accepted=True)
        )
        writer.write_footer(OptimizationReport(Path("x"), 0.5, 0.7, 1))
        content = (tmp_path / "OPTIMIZER_CHANGELOG.md").read_text()
        assert "yes" in content
        assert "Added example" in content

    def test_append_round_rejected(self, tmp_path):
        writer = ChangelogWriter(tmp_path)
        writer.write_header("s", [], 1, "2026-01-01T00:00:00Z")
        writer.append_round(
            MutationRecord(1, "Bad change", "", "", 0.5, candidate_mean=0.4, accepted=False)
        )
        writer.write_footer(OptimizationReport(Path("x"), 0.5, 0.5, 1))
        content = (tmp_path / "OPTIMIZER_CHANGELOG.md").read_text()
        assert "no" in content

    def test_write_footer_includes_scores(self, tmp_path):
        writer = ChangelogWriter(tmp_path)
        writer.write_header("s", [], 1, "2026-01-01T00:00:00Z")
        writer.write_footer(OptimizationReport(Path("x"), 0.42, 0.87, 3, converged=True))
        content = (tmp_path / "OPTIMIZER_CHANGELOG.md").read_text()
        assert "0.42" in content
        assert "0.87" in content
        assert "yes" in content.lower()

    def test_second_run_prepends(self, tmp_path):
        """Second run appears before first run in the file."""
        writer1 = ChangelogWriter(tmp_path)
        writer1.write_header("s", [], 1, "2026-01-01T00:00:00Z")
        writer1.write_footer(OptimizationReport(Path("x"), 0.5, 0.6, 1))

        writer2 = ChangelogWriter(tmp_path)
        writer2.write_header("s", [], 1, "2026-03-27T12:00:00Z")
        writer2.write_footer(OptimizationReport(Path("x"), 0.6, 0.8, 2))

        content = (tmp_path / "OPTIMIZER_CHANGELOG.md").read_text()
        pos1 = content.index("2026-01-01")
        pos2 = content.index("2026-03-27")
        assert pos2 < pos1  # newer run appears first
