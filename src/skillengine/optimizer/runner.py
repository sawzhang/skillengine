"""SkillOptimizer — Autoresearch self-improving loop for SKILL.md files.

Implements the Autoresearch hill-climbing pattern adapted for SkillEngine:

1. Score the current skill against test inputs (baseline)
2. Ask a Mutator LLM to propose ONE targeted change to the skill prompt
3. Apply the change, score again ``stability_runs`` times (stability check)
4. If mean candidate score > baseline + margin → keep, else → revert
5. Append to changelog
6. Repeat until ``pass_threshold`` reached or ``max_rounds`` exhausted

The key difference from HarnessRunner:
- HarnessRunner improves task *outputs* (ephemeral)
- SkillOptimizer improves the skill *prompt itself* (permanent)

Performance notes:
- Test inputs are scored in parallel (asyncio.gather) within each _score_skill call.
- All stability_runs fire concurrently; the mean of their scores is the acceptance gate.
- Global best tracking ensures the best-ever content is written back even if later
  rounds degrade (rare with monotonic acceptance but possible due to LLM variance).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.config import SkillsConfig
from skillengine.engine import SkillsEngine
from skillengine.events import EventBus
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

logger = logging.getLogger(__name__)


class SkillOptimizer:
    """Self-improving optimizer for SKILL.md files.

    Usage::

        optimizer = SkillOptimizer(
            config=OptimizerConfig(max_rounds=5, pass_threshold=0.85),
            base_config=AgentConfig(model="claude-sonnet-4-20250514"),
        )
        report = await optimizer.run(
            skill_path=Path("skills/my-skill/SKILL.md"),
            checklist=[
                "Output is valid JSON",
                "Response addresses the user's question",
                "No hallucinated tool names",
            ],
            test_inputs=[
                "Summarize this document",
                "List all open PRs",
            ],
        )
        print(f"Score improved: {report.initial_score:.2f} → {report.final_score:.2f}")
        print(f"Accepted mutations: {len(report.accepted_mutations)}")
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        base_config: AgentConfig | None = None,
        events: EventBus | None = None,
        engine: SkillsEngine | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.base_config = base_config or AgentConfig()
        self.events = events or EventBus()
        self.engine = engine or SkillsEngine(
            config=SkillsConfig(skill_dirs=list(self.base_config.skill_dirs))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        skill_path: Path,
        checklist: list[str],
        test_inputs: list[str],
    ) -> OptimizationReport:
        """Execute the self-improving loop.

        Args:
            skill_path: Path to the SKILL.md file to optimize.
            checklist: 3–6 evaluation criteria. Each is a short, testable statement.
            test_inputs: Sample inputs to run the skill against. More inputs = more
                reliable scoring but higher cost.

        Returns:
            OptimizationReport with scores, mutation records, and convergence status.
        """
        start_time = time.monotonic()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        changelog = ChangelogWriter(skill_path.parent, self.config.changelog_filename)
        changelog.write_header(
            skill_name=skill_path.parent.name,
            checklist=checklist,
            test_input_count=len(test_inputs),
            timestamp=timestamp,
        )

        # Baseline
        skill_content = self._read_skill(skill_path)
        baseline_score, scored_runs = await self._score_skill(skill_content, checklist, test_inputs)
        logger.info("Baseline score: %.3f", baseline_score)

        report = OptimizationReport(
            skill_path=skill_path,
            initial_score=baseline_score,
            final_score=baseline_score,
            rounds_run=0,
        )

        if baseline_score >= self.config.pass_threshold:
            report.converged = True
            logger.info("Already converged at baseline. No mutations needed.")
            changelog.write_footer(report)
            return report

        prior_mutation_descriptions: list[str] = []

        # Track the globally best content across all rounds.  With monotonic
        # acceptance this is almost always the final accepted state, but LLM
        # variance during stability scoring can occasionally cause a round to
        # accept content that is slightly worse in later evaluation.
        best_score = baseline_score
        best_content = skill_content

        for round_num in range(self.config.max_rounds):
            report.rounds_run += 1
            current_content = self._read_skill(skill_path)

            # --- Mutate ---
            new_content, mutation_desc = await self._run_mutator(
                skill_content=current_content,
                checklist=checklist,
                baseline_score=baseline_score,
                scored_runs=scored_runs,
                prior_mutation_descriptions=prior_mutation_descriptions,
            )

            # Guard: skip no-op mutations
            if new_content.strip() == current_content.strip():
                logger.warning(
                    "Round %d: mutator returned identical content — skipping",
                    round_num + 1,
                )
                continue

            # Write candidate to disk (atomic)
            self._write_skill(skill_path, new_content)

            # --- Stability scoring (all runs concurrent) ---
            candidate_scores = await self._score_stability(new_content, checklist, test_inputs)
            candidate_mean = sum(candidate_scores) / len(candidate_scores)

            accepted = candidate_mean >= baseline_score + self.config.improvement_margin

            record = MutationRecord(
                round_number=round_num + 1,
                mutation_description=mutation_desc,
                original_content=current_content,
                mutated_content=new_content,
                baseline_score=baseline_score,
                candidate_scores=candidate_scores,
                candidate_mean=candidate_mean,
                accepted=accepted,
            )
            report.mutations.append(record)
            changelog.append_round(record)
            prior_mutation_descriptions.append(mutation_desc)

            logger.info(
                "Round %d: %s  baseline=%.3f candidate=%.3f → %s",
                round_num + 1,
                mutation_desc[:60],
                baseline_score,
                candidate_mean,
                "ACCEPTED" if accepted else "REVERTED",
            )

            if accepted:
                baseline_score = candidate_mean
                report.final_score = baseline_score
                if candidate_mean > best_score:
                    best_score = candidate_mean
                    best_content = new_content
                # Refresh scored_runs so next mutator sees current weaknesses
                _, scored_runs = await self._score_skill(new_content, checklist, test_inputs)
                if baseline_score >= self.config.pass_threshold:
                    report.converged = True
                    break
            else:
                # Revert to original (atomic)
                self._write_skill(skill_path, current_content)

        # Ensure the file on disk holds the globally best content.
        # This guards against the edge case where the last accepted round
        # happened to score slightly below a previous accepted round.
        current_on_disk = self._read_skill(skill_path)
        if best_content != current_on_disk:
            logger.info(
                "Restoring globally best content (score=%.3f > current=%.3f)",
                best_score,
                baseline_score,
            )
            self._write_skill(skill_path, best_content)
            report.final_score = best_score

        # Accumulate total token usage across all mutations
        for m in report.mutations:
            report.total_token_usage += m.token_usage

        report.total_duration_seconds = time.monotonic() - start_time
        changelog.write_footer(report)

        logger.info(
            "Optimization complete: %.3f → %.3f  rounds=%d  converged=%s",
            report.initial_score,
            report.final_score,
            report.rounds_run,
            report.converged,
        )
        return report

    # ------------------------------------------------------------------
    # Core loop steps
    # ------------------------------------------------------------------

    async def _score_skill(
        self,
        skill_content: str,
        checklist: list[str],
        test_inputs: list[str],
    ) -> tuple[float, list[ScoredRun]]:
        """Run skill against all test inputs in parallel and score each output.

        Each (skill_run, scorer_run) pair is launched concurrently via
        asyncio.gather so N test inputs take roughly the same wall-clock time
        as one, bounded only by the LLM rate limit.

        Returns:
            (mean_aggregate_score, list_of_scored_runs)
        """
        if not test_inputs:
            return 0.0, []

        runs = await asyncio.gather(
            *[self._run_and_score(skill_content, inp, checklist) for inp in test_inputs]
        )
        mean_score = sum(r.aggregate_score for r in runs) / len(runs)
        return mean_score, list(runs)

    async def _run_and_score(
        self,
        skill_content: str,
        test_input: str,
        checklist: list[str],
    ) -> ScoredRun:
        """Run the skill then immediately score its output (one test input)."""
        skill_output, usage = await self._run_skill(skill_content, test_input)
        scored = await self._run_scorer(test_input, skill_output, checklist)
        scored.token_usage += usage
        return scored

    async def _run_skill(
        self,
        skill_content: str,
        test_input: str,
    ) -> tuple[str, TokenUsage]:
        """Execute the skill (as system prompt) against one test input.

        Creates a fresh AgentRunner with the skill content as system prompt,
        matching the context-reset pattern from HarnessRunner._run_phase().
        """
        agent = self._create_agent(
            system_prompt=skill_content,
            model_override=None,
            max_turns=self.base_config.max_turns,
            enable_tools=self.base_config.enable_tools,
        )
        start = time.monotonic()
        response = await agent.chat(test_input)
        elapsed = time.monotonic() - start
        logger.debug("Skill run: %.2fs  input=%r", elapsed, test_input[:50])
        return response.text_content, agent.cumulative_usage

    async def _run_scorer(
        self,
        test_input: str,
        skill_output: str,
        checklist: list[str],
    ) -> ScoredRun:
        """Run the scorer agent and parse its JSON into a ScoredRun."""
        system_prompt = format_scorer_prompt(test_input, skill_output, checklist)
        agent = self._create_agent(
            system_prompt=system_prompt,
            model_override=self.config.scorer_model,
            max_turns=self.config.scorer_max_turns,
            enable_tools=False,  # scorer is pure LLM reasoning
        )
        start = time.monotonic()
        response = await agent.chat("Score the skill output against the checklist.")
        elapsed = time.monotonic() - start

        scored = self._parse_scored_run(response.text_content, test_input, skill_output)
        scored.duration_seconds = elapsed
        scored.token_usage = agent.cumulative_usage
        return scored

    async def _run_mutator(
        self,
        skill_content: str,
        checklist: list[str],
        baseline_score: float,
        scored_runs: list[ScoredRun],
        prior_mutation_descriptions: list[str],
    ) -> tuple[str, str]:
        """Run the mutator agent. Returns (new_skill_content, mutation_description)."""
        weak_criteria = self._extract_weak_criteria(scored_runs)
        system_prompt = format_mutator_prompt(
            skill_content=skill_content,
            checklist=checklist,
            baseline_score=baseline_score,
            weak_criteria=weak_criteria,
            prior_mutations=prior_mutation_descriptions,
        )
        agent = self._create_agent(
            system_prompt=system_prompt,
            model_override=self.config.mutator_model,
            max_turns=self.config.mutator_max_turns,
            enable_tools=False,  # mutator is pure LLM reasoning
        )
        response = await agent.chat("Propose one improvement to the skill.")
        new_content, description = self._parse_mutator_output(response.text_content, skill_content)
        return new_content, description

    async def _score_stability(
        self,
        candidate_content: str,
        checklist: list[str],
        test_inputs: list[str],
    ) -> list[float]:
        """Score the candidate content stability_runs times concurrently.

        All stability runs fire in parallel via asyncio.gather so the wall-clock
        cost is that of one scoring pass (× LLM concurrency limits), not N×.
        Returns list of per-run mean scores for the acceptance gate.
        """
        results = await asyncio.gather(
            *[
                self._score_skill(candidate_content, checklist, test_inputs)
                for _ in range(self.config.stability_runs)
            ]
        )
        return [mean_score for mean_score, _ in results]

    # ------------------------------------------------------------------
    # Agent creation (identical pattern to HarnessRunner._create_agent)
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        system_prompt: str,
        model_override: str | None = None,
        max_turns: int | None = None,
        enable_tools: bool | None = None,
    ) -> AgentRunner:
        """Create a fresh AgentRunner with full context reset.

        Each call produces a brand-new agent — the context-reset mechanism
        that prevents "context anxiety" across optimizer rounds.
        """
        config = AgentConfig(
            model=model_override or self.base_config.model,
            base_url=self.base_config.base_url,
            api_key=self.base_config.api_key,
            temperature=self.base_config.temperature,
            max_tokens=self.base_config.max_tokens,
            max_turns=max_turns or self.base_config.max_turns,
            enable_tools=(
                enable_tools if enable_tools is not None else self.base_config.enable_tools
            ),
            auto_execute=self.base_config.auto_execute,
            thinking_level=self.base_config.thinking_level,
            transport=self.base_config.transport,
            skill_dirs=list(self.base_config.skill_dirs),
            system_prompt=system_prompt,
            cache_retention=self.base_config.cache_retention,
            load_context_files=False,
        )
        return AgentRunner(self.engine, config, events=self.events)

    # ------------------------------------------------------------------
    # File I/O (atomic read/write)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_skill(skill_path: Path) -> str:
        return skill_path.read_text(encoding="utf-8")

    @staticmethod
    def _write_skill(skill_path: Path, content: str) -> None:
        """Atomically write SKILL.md via a temp file + os.replace().

        Crash-safe: if the process dies mid-write, the original is untouched.
        """
        tmp_path = skill_path.parent / "SKILL.md.tmp"
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, skill_path)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scored_run(
        scorer_output: str,
        test_input: str,
        skill_output: str,
    ) -> ScoredRun:
        """Parse scorer JSON into a ScoredRun.

        Tolerates markdown fences and surrounding prose (same pattern as
        HarnessRunner._parse_evaluation).
        """
        text = scorer_output.strip()

        # Strip code fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        data: dict[str, Any] = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    data = json.loads(text[brace_start : brace_end + 1])
                except json.JSONDecodeError:
                    pass

        criterion_scores: list[CriterionScore] = []
        for item in data.get("criterion_scores", []):
            criterion_scores.append(
                CriterionScore(
                    criterion=item.get("criterion", ""),
                    passed=bool(item.get("passed", False)),
                    score=float(item.get("score", 0.0)),
                    rationale=item.get("rationale", ""),
                )
            )

        # Prefer the LLM's aggregate; fall back to computing it locally
        if criterion_scores:
            local_mean = sum(c.score for c in criterion_scores) / len(criterion_scores)
        else:
            local_mean = 0.0
        aggregate = float(data.get("aggregate_score", local_mean))

        return ScoredRun(
            test_input=test_input,
            skill_output=skill_output,
            criterion_scores=criterion_scores,
            aggregate_score=aggregate,
        )

    @staticmethod
    def _parse_mutator_output(
        mutator_output: str,
        original_content: str,
    ) -> tuple[str, str]:
        """Extract (skill_content, mutation_description) from mutator output.

        Looks for ```skill ... ``` and ```mutation_description ... ``` blocks.
        Falls back to (original_content, fallback_message) if either is missing.
        """
        text = mutator_output

        # Extract ```skill block
        skill_content: str | None = None
        if "```skill" in text:
            try:
                start = text.index("```skill") + 8
                end = text.index("```", start)
                skill_content = text[start:end].strip()
            except ValueError:
                pass

        # Extract ```mutation_description block
        description: str | None = None
        if "```mutation_description" in text:
            try:
                start = text.index("```mutation_description") + 23
                end = text.index("```", start)
                description = text[start:end].strip()
            except ValueError:
                pass

        if skill_content is None or description is None:
            logger.warning("Mutator output missing required blocks — treating as no-op")
            return original_content, "mutator produced no parseable output"

        return skill_content, description

    @staticmethod
    def _extract_weak_criteria(
        scored_runs: list[ScoredRun],
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """Return the top_n lowest-scoring criteria by mean score across all runs.

        Used to focus the mutator on what matters most.
        """
        if not scored_runs:
            return []

        # Collect all criterion names
        all_criteria: set[str] = set()
        for run in scored_runs:
            for cs in run.criterion_scores:
                all_criteria.add(cs.criterion)

        # Mean score per criterion across all runs
        criterion_means: dict[str, float] = {}
        for criterion in all_criteria:
            scores = []
            for run in scored_runs:
                for cs in run.criterion_scores:
                    if cs.criterion == criterion:
                        scores.append(cs.score)
            if scores:
                criterion_means[criterion] = sum(scores) / len(scores)

        # Sort ascending (weakest first), return top_n
        sorted_criteria = sorted(criterion_means.items(), key=lambda x: x[1])
        return sorted_criteria[:top_n]
