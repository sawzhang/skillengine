"""HarnessRunner — multi-agent orchestrator for long-running tasks.

Implements the Planner→Generator→Evaluator harness pattern from Anthropic's
research on harness design for long-running applications. Builds on top of
AgentRunner without replacing it.

Three tiers of complexity:
- Tier 1: Generator + Evaluator refinement loop
- Tier 2: + Planner that expands user prompt into detailed spec
- Tier 3: + Sprint decomposition with context resets per sprint
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.config import SkillsConfig
from skillengine.engine import SkillsEngine
from skillengine.events import EventBus
from skillengine.harness.events import (
    EVALUATION_COMPLETE,
    HARNESS_END,
    HARNESS_START,
    PHASE_END,
    PHASE_START,
    SPRINT_END,
    SPRINT_START,
    EvaluationCompleteEvent,
    HarnessEndEvent,
    HarnessStartEvent,
    PhaseEndEvent,
    PhaseStartEvent,
    SprintEndEvent,
    SprintStartEvent,
)
from skillengine.harness.models import (
    EvaluationResult,
    HarnessConfig,
    HarnessReport,
    PhaseResult,
    SprintContract,
    SprintResult,
)
from skillengine.harness.prompts import (
    format_evaluator_prompt,
    format_generator_prompt,
    format_planner_prompt,
)

logger = logging.getLogger(__name__)


class HarnessRunner:
    """Multi-agent harness orchestrator.

    Composes multiple AgentRunner instances with context resets,
    sprint contracts, and iterative refinement.

    Example::

        harness = HarnessRunner(
            config=HarnessConfig(enable_planner=True),
            base_config=AgentConfig(model="claude-sonnet-4-20250514"),
        )
        report = await harness.run("Build a REST API for user management")
        print(report.cost_breakdown_by_phase())
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        base_config: AgentConfig | None = None,
        events: EventBus | None = None,
        engine: SkillsEngine | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.base_config = base_config or AgentConfig()
        self.events = events or EventBus()
        self.engine = engine or SkillsEngine(
            config=SkillsConfig(skill_dirs=list(self.base_config.skill_dirs))
        )

    async def run(self, user_input: str) -> HarnessReport:
        """Execute the harness pipeline.

        Selects tier based on config:
        - enable_sprints → Tier 3
        - enable_planner → Tier 2
        - else → Tier 1
        """
        start_time = time.monotonic()
        await self.events.emit(
            HARNESS_START,
            HarnessStartEvent(config=self.config, user_input=user_input),
        )

        try:
            if self.config.enable_sprints and self.config.enable_planner:
                report = await self._run_with_sprints(user_input)
            elif self.config.enable_planner:
                report = await self._run_with_planner(user_input)
            else:
                report = await self._run_tier1(user_input)
        except Exception:
            logger.exception("Harness run failed")
            raise
        finally:
            elapsed = time.monotonic() - start_time
            if "report" in locals():
                report.total_duration_seconds = elapsed
                await self.events.emit(HARNESS_END, HarnessEndEvent(report=report))

        return report

    # ------------------------------------------------------------------
    # Tier 1: Generator + Evaluator
    # ------------------------------------------------------------------

    async def _run_tier1(self, user_input: str) -> HarnessReport:
        """Tier 1: single Generator+Evaluator loop with user input as spec."""
        # Use the user input directly as task + criteria
        criteria = [f"The implementation fully addresses the following request: {user_input}"]
        sprint = await self._run_generator_evaluator(
            task_description=user_input,
            acceptance_criteria=criteria,
        )
        return HarnessReport(sprints=[sprint])

    async def _run_generator_evaluator(
        self,
        task_description: str,
        acceptance_criteria: list[str],
        prior_work_summary: str = "",
        sprint_number: int = 0,
    ) -> SprintResult:
        """Core Generator↔Evaluator refinement loop.

        Each iteration creates FRESH agents (context reset).
        """
        contract = SprintContract(
            sprint_number=sprint_number,
            title=task_description[:80],
            description=task_description,
            acceptance_criteria=acceptance_criteria,
        )
        result = SprintResult(contract=contract)
        evaluator_feedback = ""

        for round_num in range(self.config.max_refinement_rounds):
            result.total_refinement_rounds = round_num + 1

            # --- Generator phase ---
            gen_result = await self._run_phase(
                phase="generator",
                system_prompt=format_generator_prompt(
                    task_description=task_description,
                    acceptance_criteria=acceptance_criteria,
                    prior_work_summary=prior_work_summary,
                    evaluator_feedback=evaluator_feedback,
                ),
                user_message="Begin implementation.",
                model_override=self.config.generator_model,
                max_turns=self.config.generator_max_turns,
                tool_filter=self.config.generator_tools,
                sprint_number=sprint_number,
                refinement_round=round_num,
            )
            result.phases.append(gen_result)

            # --- Evaluator phase ---
            eval_result = await self._run_phase(
                phase="evaluator",
                system_prompt=format_evaluator_prompt(acceptance_criteria),
                user_message="Evaluate the current state of the implementation.",
                model_override=self.config.evaluator_model,
                max_turns=self.config.evaluator_max_turns,
                tool_filter=self.config.evaluator_tools,
                sprint_number=sprint_number,
                refinement_round=round_num,
            )
            evaluation = self._parse_evaluation(eval_result.output)
            eval_result.evaluation = evaluation
            result.phases.append(eval_result)

            await self.events.emit(
                EVALUATION_COMPLETE,
                EvaluationCompleteEvent(
                    sprint_number=sprint_number,
                    refinement_round=round_num,
                    result=evaluation,
                ),
            )

            logger.info(
                "Sprint %d Round %d: score=%.2f passed=%s",
                sprint_number,
                round_num,
                evaluation.score,
                evaluation.passed,
            )

            if evaluation.passed and evaluation.score >= self.config.pass_threshold:
                result.final_passed = True
                break

            # Carry feedback to next round
            evaluator_feedback = evaluation.feedback
            if evaluation.suggestions:
                evaluator_feedback += "\n\nSpecific suggestions:\n" + "\n".join(
                    f"- {s}" for s in evaluation.suggestions
                )

        return result

    # ------------------------------------------------------------------
    # Tier 2: Planner + Generator + Evaluator
    # ------------------------------------------------------------------

    async def _run_with_planner(self, user_input: str) -> HarnessReport:
        """Tier 2: Planner expands prompt, then single Generator+Evaluator."""
        planner_result = await self._run_phase(
            phase="planner",
            system_prompt=format_planner_prompt(user_input),
            user_message="Create the implementation plan.",
            model_override=self.config.planner_model,
            max_turns=self.config.planner_max_turns,
            sprint_number=0,
            refinement_round=0,
        )

        plan = self._parse_planner_output(planner_result.output)
        criteria = plan.get("acceptance_criteria", [])
        summary = plan.get("project_summary", user_input)

        if not criteria:
            criteria = [f"Implement: {user_input}"]

        sprint = await self._run_generator_evaluator(
            task_description=summary,
            acceptance_criteria=criteria,
        )
        return HarnessReport(sprints=[sprint], planner_result=planner_result)

    # ------------------------------------------------------------------
    # Tier 3: Planner + Sprints + Generator + Evaluator
    # ------------------------------------------------------------------

    async def _run_with_sprints(self, user_input: str) -> HarnessReport:
        """Tier 3: Planner decomposes into sprints, each gets its own loop."""
        planner_result = await self._run_phase(
            phase="planner",
            system_prompt=format_planner_prompt(user_input),
            user_message="Create the implementation plan with sprint decomposition.",
            model_override=self.config.planner_model,
            max_turns=self.config.planner_max_turns,
            sprint_number=0,
            refinement_round=0,
        )

        contracts = self._parse_sprint_contracts(planner_result.output)
        if not contracts:
            contracts = [
                SprintContract(
                    sprint_number=1,
                    title="Full implementation",
                    description=user_input,
                    acceptance_criteria=[f"Implement: {user_input}"],
                )
            ]

        # Cap at max_sprints
        contracts = contracts[: self.config.max_sprints]

        report = HarnessReport(planner_result=planner_result)
        completed_sprints: list[SprintResult] = []

        for contract in contracts:
            await self.events.emit(
                SPRINT_START,
                SprintStartEvent(sprint_number=contract.sprint_number, contract=contract),
            )

            prior_summary = self._build_context_handoff(completed_sprints)

            sprint_result = await self._run_generator_evaluator(
                task_description=contract.description,
                acceptance_criteria=contract.acceptance_criteria,
                prior_work_summary=prior_summary,
                sprint_number=contract.sprint_number,
            )
            sprint_result.contract = contract
            report.sprints.append(sprint_result)
            completed_sprints.append(sprint_result)

            await self.events.emit(
                SPRINT_END,
                SprintEndEvent(sprint_number=contract.sprint_number, result=sprint_result),
            )

        return report

    # ------------------------------------------------------------------
    # Agent creation (follows _execute_skill_forked pattern)
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        system_prompt: str,
        model_override: str | None = None,
        max_turns: int | None = None,
        tool_filter: list[str] | None = None,
    ) -> AgentRunner:
        """Create a child AgentRunner with fresh context.

        Each call produces a brand-new agent (context reset) — the key
        mechanism for solving context anxiety in long-running tasks.
        """
        config = AgentConfig(
            model=model_override or self.base_config.model,
            base_url=self.base_config.base_url,
            api_key=self.base_config.api_key,
            temperature=self.base_config.temperature,
            max_tokens=self.base_config.max_tokens,
            max_turns=max_turns or self.base_config.max_turns,
            enable_tools=self.base_config.enable_tools,
            auto_execute=self.base_config.auto_execute,
            thinking_level=self.base_config.thinking_level,
            transport=self.base_config.transport,
            skill_dirs=list(self.base_config.skill_dirs),
            system_prompt=system_prompt,
            cache_retention=self.base_config.cache_retention,
            load_context_files=False,
        )
        agent = AgentRunner(self.engine, config, events=self.events)

        # Apply tool filter if specified
        if tool_filter is not None:
            original_get_tools = agent.get_tools
            allowed = set(tool_filter) | {"execute", "execute_script"}

            def filtered_tools() -> list[dict[str, Any]]:
                return [
                    t
                    for t in original_get_tools()
                    if t.get("function", {}).get("name", "") in allowed
                ]

            agent.get_tools = filtered_tools  # type: ignore[method-assign]

        return agent

    async def _run_phase(
        self,
        phase: str,
        system_prompt: str,
        user_message: str,
        model_override: str | None = None,
        max_turns: int | None = None,
        tool_filter: list[str] | None = None,
        sprint_number: int = 0,
        refinement_round: int = 0,
    ) -> PhaseResult:
        """Run a single harness phase and capture results."""
        await self.events.emit(
            PHASE_START,
            PhaseStartEvent(
                phase=phase,
                sprint_number=sprint_number,
                refinement_round=refinement_round,
            ),
        )

        start_time = time.monotonic()
        agent = self._create_agent(
            system_prompt=system_prompt,
            model_override=model_override,
            max_turns=max_turns,
            tool_filter=tool_filter,
        )

        response = await agent.chat(user_message)
        elapsed = time.monotonic() - start_time

        result = PhaseResult(
            phase=phase,
            sprint_number=sprint_number,
            refinement_round=refinement_round,
            output=response.text_content,
            token_usage=agent.cumulative_usage,
            duration_seconds=elapsed,
        )

        await self.events.emit(PHASE_END, PhaseEndEvent(phase=phase, result=result))
        return result

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_evaluation(evaluator_output: str) -> EvaluationResult:
        """Parse the Evaluator's JSON output into an EvaluationResult.

        Tolerates markdown code fences and extra text around the JSON.
        """
        text = evaluator_output.strip()

        # Strip markdown code fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # Try to find a JSON object in the text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    data = json.loads(text[brace_start : brace_end + 1])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        return EvaluationResult(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            criteria_results=data.get("criteria_results", {}),
            feedback=data.get("feedback", evaluator_output),
            suggestions=data.get("suggestions", []),
        )

    @staticmethod
    def _parse_planner_output(planner_output: str) -> dict[str, Any]:
        """Parse the Planner's JSON output."""
        text = planner_output.strip()
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    return json.loads(text[brace_start : brace_end + 1])
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _parse_sprint_contracts(planner_output: str) -> list[SprintContract]:
        """Parse Planner output into a list of SprintContracts."""
        data = HarnessRunner._parse_planner_output(planner_output)
        sprints_raw = data.get("sprints", [])
        contracts: list[SprintContract] = []
        for i, s in enumerate(sprints_raw):
            contracts.append(
                SprintContract(
                    sprint_number=s.get("sprint_number", i + 1),
                    title=s.get("title", f"Sprint {i + 1}"),
                    description=s.get("description", ""),
                    acceptance_criteria=s.get("acceptance_criteria", []),
                    estimated_complexity=s.get("estimated_complexity", "medium"),
                )
            )
        return contracts

    @staticmethod
    def _build_context_handoff(completed_sprints: list[SprintResult]) -> str:
        """Build a structured summary of completed sprints.

        This is the key to solving context anxiety: each new sprint gets
        a compact summary instead of the full conversation history.
        """
        if not completed_sprints:
            return ""

        lines = ["## Completed Work\n"]
        for sprint in completed_sprints:
            status = "PASSED" if sprint.final_passed else "INCOMPLETE"
            lines.append(
                f"### Sprint {sprint.contract.sprint_number}: {sprint.contract.title} [{status}]"
            )
            lines.append(f"\n{sprint.contract.description}\n")

            # Include the last generator output as the most relevant summary
            gen_phases = [p for p in sprint.phases if p.phase == "generator"]
            if gen_phases:
                last_gen = gen_phases[-1]
                # Truncate to keep handoff compact
                summary = last_gen.output
                if len(summary) > 2000:
                    summary = summary[:2000] + "\n\n[... truncated]"
                lines.append(f"Implementation summary:\n{summary}\n")

        return "\n".join(lines)
