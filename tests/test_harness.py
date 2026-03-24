"""Tests for the harness multi-agent orchestration system."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
from skillengine.events import EventBus
from skillengine.harness.events import (
    EVALUATION_COMPLETE,
    HARNESS_END,
    HARNESS_START,
    PHASE_END,
    PHASE_START,
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
from skillengine.harness.runner import HarnessRunner
from skillengine.model_registry import TokenUsage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_message(content: str) -> AgentMessage:
    return AgentMessage(role="assistant", content=content)


def _make_evaluation_json(
    passed: bool = True,
    score: float = 0.9,
    criteria: dict[str, bool] | None = None,
    feedback: str = "Good work.",
    suggestions: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "passed": passed,
            "score": score,
            "criteria_results": criteria or {"requirement met": True},
            "feedback": feedback,
            "suggestions": suggestions or [],
        }
    )


def _make_planner_json(
    summary: str = "Build the thing",
    criteria: list[str] | None = None,
    sprints: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "project_summary": summary,
            "acceptance_criteria": criteria or ["It works"],
            "sprints": sprints
            or [
                {
                    "sprint_number": 1,
                    "title": "Core",
                    "description": "Build core",
                    "acceptance_criteria": ["Core works"],
                    "estimated_complexity": "medium",
                }
            ],
        }
    )


@pytest.fixture
def harness_config():
    return HarnessConfig(max_refinement_rounds=3)


@pytest.fixture
def base_config():
    return AgentConfig(model="test-model", enable_tools=False)


@pytest.fixture
def event_bus():
    return EventBus()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestHarnessModels:
    def test_sprint_result_total_usage(self):
        sr = SprintResult(
            contract=SprintContract(
                sprint_number=1,
                title="Test",
                description="Test sprint",
                acceptance_criteria=["works"],
            ),
            phases=[
                PhaseResult(
                    phase="generator",
                    token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                ),
                PhaseResult(
                    phase="evaluator",
                    token_usage=TokenUsage(input_tokens=80, output_tokens=30),
                ),
            ],
        )
        usage = sr.total_usage
        assert usage.input_tokens == 180
        assert usage.output_tokens == 80

    def test_harness_report_total_usage(self):
        report = HarnessReport(
            planner_result=PhaseResult(
                phase="planner",
                token_usage=TokenUsage(input_tokens=50, output_tokens=20),
            ),
            sprints=[
                SprintResult(
                    contract=SprintContract(
                        sprint_number=1,
                        title="T",
                        description="D",
                        acceptance_criteria=[],
                    ),
                    phases=[
                        PhaseResult(
                            phase="generator",
                            token_usage=TokenUsage(
                                input_tokens=200, output_tokens=100
                            ),
                        ),
                    ],
                )
            ],
        )
        usage = report.total_usage
        assert usage.input_tokens == 250
        assert usage.output_tokens == 120

    def test_cost_breakdown_by_phase(self):
        report = HarnessReport(
            planner_result=PhaseResult(
                phase="planner",
                token_usage=TokenUsage(input_tokens=50),
            ),
            sprints=[
                SprintResult(
                    contract=SprintContract(
                        sprint_number=1,
                        title="T",
                        description="D",
                        acceptance_criteria=[],
                    ),
                    phases=[
                        PhaseResult(
                            phase="generator",
                            sprint_number=1,
                            refinement_round=0,
                            token_usage=TokenUsage(input_tokens=200),
                        ),
                        PhaseResult(
                            phase="evaluator",
                            sprint_number=1,
                            refinement_round=0,
                            token_usage=TokenUsage(input_tokens=80),
                        ),
                    ],
                )
            ],
        )
        breakdown = report.cost_breakdown_by_phase()
        assert "planner" in breakdown
        assert "generator_S1_R0" in breakdown
        assert "evaluator_S1_R0" in breakdown
        assert breakdown["planner"].input_tokens == 50
        assert breakdown["generator_S1_R0"].input_tokens == 200


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_generator_prompt_basic(self):
        prompt = format_generator_prompt(
            task_description="Build a REST API",
            acceptance_criteria=["Endpoints work", "Tests pass"],
        )
        assert "Build a REST API" in prompt
        assert "- Endpoints work" in prompt
        assert "- Tests pass" in prompt
        assert "Prior Work" not in prompt
        assert "Evaluator Feedback" not in prompt

    def test_generator_prompt_with_feedback(self):
        prompt = format_generator_prompt(
            task_description="Build API",
            acceptance_criteria=["Works"],
            evaluator_feedback="The POST endpoint returns 500.",
        )
        assert "Evaluator Feedback" in prompt
        assert "POST endpoint returns 500" in prompt

    def test_generator_prompt_with_prior_work(self):
        prompt = format_generator_prompt(
            task_description="Build API",
            acceptance_criteria=["Works"],
            prior_work_summary="Sprint 1 completed: database schema created.",
        )
        assert "Prior Work" in prompt
        assert "database schema created" in prompt

    def test_evaluator_prompt(self):
        prompt = format_evaluator_prompt(["API returns 200", "Tests pass"])
        assert "- API returns 200" in prompt
        assert "- Tests pass" in prompt
        assert "JSON" in prompt

    def test_planner_prompt(self):
        prompt = format_planner_prompt("Build a todo app")
        assert "Build a todo app" in prompt
        assert "sprints" in prompt


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parse_evaluation_clean_json(self):
        result = HarnessRunner._parse_evaluation(
            _make_evaluation_json(passed=True, score=0.95)
        )
        assert result.passed is True
        assert result.score == 0.95

    def test_parse_evaluation_with_code_fence(self):
        text = "Here is my evaluation:\n```json\n" + _make_evaluation_json() + "\n```"
        result = HarnessRunner._parse_evaluation(text)
        assert result.passed is True

    def test_parse_evaluation_with_surrounding_text(self):
        text = (
            "I tested everything.\n"
            + _make_evaluation_json(passed=False, score=0.3, feedback="Broken")
            + "\nThat's my assessment."
        )
        result = HarnessRunner._parse_evaluation(text)
        assert result.passed is False
        assert result.score == 0.3
        assert result.feedback == "Broken"

    def test_parse_evaluation_garbage_input(self):
        result = HarnessRunner._parse_evaluation("This is not JSON at all.")
        assert result.passed is False
        assert result.score == 0.0

    def test_parse_sprint_contracts(self):
        planner_output = _make_planner_json(
            sprints=[
                {
                    "sprint_number": 1,
                    "title": "Auth",
                    "description": "Build auth",
                    "acceptance_criteria": ["Login works"],
                },
                {
                    "sprint_number": 2,
                    "title": "API",
                    "description": "Build API",
                    "acceptance_criteria": ["CRUD works"],
                },
            ]
        )
        contracts = HarnessRunner._parse_sprint_contracts(planner_output)
        assert len(contracts) == 2
        assert contracts[0].title == "Auth"
        assert contracts[1].acceptance_criteria == ["CRUD works"]

    def test_parse_planner_output_with_code_fence(self):
        text = "```json\n" + _make_planner_json(summary="My plan") + "\n```"
        data = HarnessRunner._parse_planner_output(text)
        assert data["project_summary"] == "My plan"


# ---------------------------------------------------------------------------
# Context handoff tests
# ---------------------------------------------------------------------------


class TestContextHandoff:
    def test_empty_handoff(self):
        assert HarnessRunner._build_context_handoff([]) == ""

    def test_single_sprint_handoff(self):
        sr = SprintResult(
            contract=SprintContract(
                sprint_number=1,
                title="Auth",
                description="Build authentication",
                acceptance_criteria=["Login works"],
            ),
            final_passed=True,
            phases=[
                PhaseResult(
                    phase="generator", output="Implemented JWT login with bcrypt."
                ),
            ],
        )
        handoff = HarnessRunner._build_context_handoff([sr])
        assert "Sprint 1" in handoff
        assert "Auth" in handoff
        assert "PASSED" in handoff
        assert "JWT login" in handoff

    def test_handoff_truncates_long_output(self):
        sr = SprintResult(
            contract=SprintContract(
                sprint_number=1,
                title="Big",
                description="Lots of work",
                acceptance_criteria=[],
            ),
            phases=[
                PhaseResult(phase="generator", output="x" * 5000),
            ],
        )
        handoff = HarnessRunner._build_context_handoff([sr])
        assert "[... truncated]" in handoff
        assert len(handoff) < 5000


# ---------------------------------------------------------------------------
# Runner integration tests (mock LLM)
# ---------------------------------------------------------------------------


def _mock_create_agent(chat_responses: list[AgentMessage]):
    """Create a patched _create_agent that returns mock agents with queued responses."""
    call_index = [0]

    def factory(self, system_prompt, model_override=None, max_turns=None, tool_filter=None):
        agent = MagicMock(spec=AgentRunner)
        agent._cumulative_usage = TokenUsage()
        agent.cumulative_usage = TokenUsage()

        idx = call_index[0]
        if idx < len(chat_responses):
            agent.chat = AsyncMock(return_value=chat_responses[idx])
            call_index[0] += 1
        else:
            agent.chat = AsyncMock(return_value=_make_agent_message(""))

        return agent

    return factory


class TestHarnessRunnerTier1:
    """Tier 1: Generator + Evaluator."""

    async def test_happy_path_passes_first_round(self, harness_config, base_config):
        """Generator output passes evaluation on first try."""
        responses = [
            _make_agent_message("I implemented everything."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.95)),
        ]

        harness = HarnessRunner(config=harness_config, base_config=base_config)
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            report = await harness.run("Build a hello world")

        assert len(report.sprints) == 1
        assert report.sprints[0].final_passed is True
        assert report.sprints[0].total_refinement_rounds == 1

    async def test_refinement_loop(self, harness_config, base_config):
        """Evaluator fails first round, passes second."""
        responses = [
            _make_agent_message("First attempt."),
            _make_agent_message(
                _make_evaluation_json(
                    passed=False, score=0.4, feedback="Missing feature X"
                )
            ),
            _make_agent_message("Added feature X."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.9)),
        ]

        harness = HarnessRunner(config=harness_config, base_config=base_config)
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            report = await harness.run("Build feature X")

        assert report.sprints[0].final_passed is True
        assert report.sprints[0].total_refinement_rounds == 2

    async def test_max_rounds_exceeded(self, base_config):
        """Evaluator never passes, stops at max rounds."""
        config = HarnessConfig(max_refinement_rounds=2)
        responses = [
            _make_agent_message("Attempt 1."),
            _make_agent_message(_make_evaluation_json(passed=False, score=0.3)),
            _make_agent_message("Attempt 2."),
            _make_agent_message(_make_evaluation_json(passed=False, score=0.5)),
        ]

        harness = HarnessRunner(config=config, base_config=base_config)
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            report = await harness.run("Impossible task")

        assert report.sprints[0].final_passed is False
        assert report.sprints[0].total_refinement_rounds == 2


class TestHarnessRunnerTier2:
    """Tier 2: Planner + Generator + Evaluator."""

    async def test_planner_expands_prompt(self, base_config):
        config = HarnessConfig(enable_planner=True)
        responses = [
            _make_agent_message(
                _make_planner_json(
                    summary="Build a REST API",
                    criteria=["GET /users returns 200", "POST /users creates user"],
                )
            ),
            _make_agent_message("Built the API."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.9)),
        ]

        harness = HarnessRunner(config=config, base_config=base_config)
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            report = await harness.run("Build API")

        assert report.planner_result is not None
        assert report.planner_result.phase == "planner"
        assert report.sprints[0].final_passed is True


class TestHarnessRunnerTier3:
    """Tier 3: Planner + Sprints + Generator + Evaluator."""

    async def test_multi_sprint_execution(self, base_config):
        config = HarnessConfig(enable_planner=True, enable_sprints=True)
        responses = [
            _make_agent_message(
                _make_planner_json(
                    sprints=[
                        {
                            "sprint_number": 1,
                            "title": "Auth",
                            "description": "Build auth",
                            "acceptance_criteria": ["Login works"],
                        },
                        {
                            "sprint_number": 2,
                            "title": "API",
                            "description": "Build API",
                            "acceptance_criteria": ["CRUD works"],
                        },
                    ]
                )
            ),
            _make_agent_message("Built auth."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.9)),
            _make_agent_message("Built API."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.85)),
        ]

        harness = HarnessRunner(config=config, base_config=base_config)
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            report = await harness.run("Build a full app")

        assert report.planner_result is not None
        assert len(report.sprints) == 2
        assert report.sprints[0].contract.title == "Auth"
        assert report.sprints[1].contract.title == "API"
        assert all(s.final_passed for s in report.sprints)


# ---------------------------------------------------------------------------
# Event emission tests
# ---------------------------------------------------------------------------


class TestHarnessEvents:
    async def test_events_emitted_in_order(self, base_config):
        responses = [
            _make_agent_message("Done."),
            _make_agent_message(_make_evaluation_json(passed=True, score=0.9)),
        ]

        bus = EventBus()
        events_received: list[str] = []

        bus.on(HARNESS_START, lambda e: events_received.append("harness_start"))
        bus.on(PHASE_START, lambda e: events_received.append(f"phase_start:{e.phase}"))
        bus.on(PHASE_END, lambda e: events_received.append(f"phase_end:{e.phase}"))
        bus.on(
            EVALUATION_COMPLETE, lambda e: events_received.append("evaluation_complete")
        )
        bus.on(HARNESS_END, lambda e: events_received.append("harness_end"))

        harness = HarnessRunner(
            config=HarnessConfig(), base_config=base_config, events=bus
        )
        with patch.object(
            HarnessRunner, "_create_agent", _mock_create_agent(responses)
        ):
            await harness.run("Test")

        assert events_received == [
            "harness_start",
            "phase_start:generator",
            "phase_end:generator",
            "phase_start:evaluator",
            "phase_end:evaluator",
            "evaluation_complete",
            "harness_end",
        ]
