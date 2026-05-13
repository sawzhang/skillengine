"""Tests for EVAL-1: eval harness — dataset, scorers, runner, suites, CLI."""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillengine import (
    ContainsScorer,
    EvalCase,
    EvalDataset,
    EvalRunner,
    ExactMatchScorer,
    LLMJudgeScorer,
    RegexScorer,
    Scorer,
    ScorerResult,
    StructuredMatchScorer,
    builtin_suite,
    list_builtin_suites,
)

# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def test_dataset_jsonl_roundtrip(tmp_path) -> None:
    ds = EvalDataset(
        name="x",
        cases=[
            EvalCase(id="a", input="hi", expected="hello", tags=["smoke"]),
            EvalCase(id="b", input=42, expected=43),
        ],
    )
    p = tmp_path / "x.jsonl"
    ds.to_jsonl(p)
    loaded = EvalDataset.from_jsonl(p)
    assert loaded.name == "x"
    assert len(loaded) == 2
    assert loaded.cases[0].id == "a"
    assert loaded.cases[1].input == 42


def test_dataset_json_roundtrip(tmp_path) -> None:
    ds = EvalDataset(
        name="y", description="desc", cases=[EvalCase(id="1", input="x", expected="y")]
    )
    p = tmp_path / "y.json"
    ds.to_json(p)
    loaded = EvalDataset.from_json(p)
    assert loaded.name == "y"
    assert loaded.description == "desc"
    assert loaded.cases[0].expected == "y"


def test_dataset_filter_by_tag_and_id() -> None:
    ds = EvalDataset(
        name="d",
        cases=[
            EvalCase(id="a", input=1, tags=["smoke"]),
            EvalCase(id="b", input=2, tags=["slow"]),
            EvalCase(id="c", input=3, tags=["smoke", "slow"]),
        ],
    )
    smoke = ds.filter(tags=["smoke"])
    assert [c.id for c in smoke] == ["a", "c"]
    only_b = ds.filter(ids=["b"])
    assert [c.id for c in only_b] == ["b"]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_match_scorer() -> None:
    s = ExactMatchScorer()
    assert (await s.score("hello", "hello")).passed
    assert not (await s.score("hello", "world")).passed
    # strip + case
    s2 = ExactMatchScorer(case_insensitive=True, strip=True)
    assert (await s2.score("  HELLO\n", "hello")).passed


@pytest.mark.asyncio
async def test_contains_scorer_string_and_list() -> None:
    s = ContainsScorer()
    assert (await s.score("the quick fox", "fox")).passed
    multi = await s.score("apple, banana, cherry", ["apple", "banana"])
    assert multi.passed and multi.score == 1.0
    missing = await s.score("only apple", ["apple", "banana"])
    assert not missing.passed
    assert missing.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_regex_scorer() -> None:
    s = RegexScorer()
    assert (await s.score("file-123.txt", r"file-\d+")).passed
    assert not (await s.score("file.txt", r"file-\d+")).passed
    full = RegexScorer(fullmatch=True)
    assert not (await full.score("file-123.txt suffix", r"file-\d+\.txt")).passed
    assert (await full.score("file-123.txt", r"file-\d+\.txt")).passed


@pytest.mark.asyncio
async def test_structured_match_scorer_subset_and_strict() -> None:
    s = StructuredMatchScorer()
    actual = {"name": "alice", "age": 30, "extra": "ok"}
    assert (await s.score(actual, {"name": "alice", "age": 30})).passed
    # extra keys allowed by default
    assert (await s.score(actual, {"name": "alice"})).passed
    # strict mode rejects extras
    strict = StructuredMatchScorer(strict=True)
    assert not (await strict.score(actual, {"name": "alice"})).passed


@pytest.mark.asyncio
async def test_structured_match_scorer_parses_json_string() -> None:
    s = StructuredMatchScorer()
    result = await s.score('{"x": 1, "y": 2}', {"x": 1})
    assert result.passed
    bad = await s.score("not json", {"x": 1})
    assert not bad.passed


@pytest.mark.asyncio
async def test_structured_match_scorer_nested_list() -> None:
    s = StructuredMatchScorer()
    actual = {"steps": ["a", "b", "c"]}
    assert (await s.score(actual, {"steps": ["a", "b", "c"]})).passed
    assert not (await s.score(actual, {"steps": ["a", "b"]})).passed  # length mismatch


@pytest.mark.asyncio
async def test_llm_judge_scorer_pass_and_fail() -> None:
    async def judge(prompt: str) -> str:
        if "good" in prompt:
            return "PASS the actual answer matches"
        return "FAIL nope"

    s = LLMJudgeScorer(judge=judge, criterion="answer is correct")
    p = await s.score("good answer", "expected")
    assert p.passed and p.score == 1.0
    f = await s.score("bad answer", "expected")
    assert not f.passed


@pytest.mark.asyncio
async def test_llm_judge_scorer_no_judge_fails() -> None:
    s = LLMJudgeScorer()
    r = await s.score("x", "y")
    assert not r.passed
    assert "no judge" in r.reason


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_passes_simple_sync_target() -> None:
    ds = EvalDataset(
        name="t",
        cases=[
            EvalCase(id="a", input="hello", expected="hello"),
            EvalCase(id="b", input="world", expected="world"),
        ],
    )
    runner = EvalRunner(target=lambda x: x, scorers=[ExactMatchScorer()])
    report = await runner.run(ds)
    assert report.total == 2 and report.passed == 2
    assert report.pass_rate == 1.0
    assert report.failed == 0
    assert report.mean_score == 1.0


@pytest.mark.asyncio
async def test_runner_handles_async_target() -> None:
    async def target(x: Any) -> str:
        return f"got:{x}"

    ds = EvalDataset(
        name="t",
        cases=[EvalCase(id="a", input="hi", expected="got:hi")],
    )
    runner = EvalRunner(target=target, scorers=[ExactMatchScorer()])
    report = await runner.run(ds)
    assert report.passed == 1


@pytest.mark.asyncio
async def test_runner_records_target_exceptions() -> None:
    def boom(x: Any) -> str:
        raise RuntimeError("nope")

    ds = EvalDataset(name="t", cases=[EvalCase(id="a", input=1, expected=1)])
    runner = EvalRunner(target=boom, scorers=[ExactMatchScorer()])
    report = await runner.run(ds)
    assert report.passed == 0
    assert report.errored == 1
    assert "RuntimeError" in report.cases[0].error


@pytest.mark.asyncio
async def test_runner_records_scorer_exceptions() -> None:
    class _Boom(Scorer):
        name = "boom"

        async def score(self, actual: Any, case_expected: Any) -> ScorerResult:
            raise ValueError("scorer-down")

    ds = EvalDataset(name="t", cases=[EvalCase(id="a", input=1, expected=1)])
    runner = EvalRunner(target=lambda x: x, scorers=[_Boom()])
    report = await runner.run(ds)
    assert not report.cases[0].passed
    assert "scorer-down" in report.cases[0].scores[0].reason


@pytest.mark.asyncio
async def test_runner_requires_scorers() -> None:
    with pytest.raises(ValueError):
        EvalRunner(target=lambda x: x, scorers=[])


@pytest.mark.asyncio
async def test_runner_on_case_callback() -> None:
    seen: list[str] = []
    ds = EvalDataset(
        name="t",
        cases=[
            EvalCase(id="a", input=1, expected=1),
            EvalCase(id="b", input=2, expected=2),
        ],
    )
    runner = EvalRunner(
        target=lambda x: x,
        scorers=[ExactMatchScorer()],
        on_case=lambda r: seen.append(r.case_id),
    )
    await runner.run(ds)
    assert set(seen) == {"a", "b"}


@pytest.mark.asyncio
async def test_report_summary_and_save(tmp_path) -> None:
    ds = EvalDataset(
        name="t",
        cases=[
            EvalCase(id="a", input=1, expected=1),
            EvalCase(id="b", input=2, expected=99),
        ],
    )
    runner = EvalRunner(target=lambda x: x, scorers=[ExactMatchScorer()])
    report = await runner.run(ds)
    summary = report.summary()
    assert "1/2 passed" in summary
    p = tmp_path / "report.json"
    report.save(p)
    data = json.loads(p.read_text())
    assert data["total"] == 2
    assert data["passed"] == 1
    assert data["pass_rate"] == 0.5


# ---------------------------------------------------------------------------
# Built-in suites
# ---------------------------------------------------------------------------


def test_list_builtin_suites_includes_skill_dsl() -> None:
    suites = list_builtin_suites()
    assert "skill-dsl" in suites
    assert "smoke" in suites


@pytest.mark.asyncio
async def test_builtin_skill_dsl_suite_runs_clean() -> None:
    dataset, scorers, target = builtin_suite("skill-dsl")
    assert len(dataset) >= 30, f"skill-dsl should have ≥30 cases, got {len(dataset)}"
    runner = EvalRunner(target=target, scorers=scorers, concurrency=4)
    report = await runner.run(dataset)
    # Every built-in case is curated; the suite must pass cleanly.
    assert report.passed == report.total, (
        f"{report.failed} skill-dsl cases failed: "
        + ", ".join(r.case_id for r in report.cases if not r.passed)[:200]
    )


@pytest.mark.asyncio
async def test_builtin_smoke_suite_runs_clean() -> None:
    dataset, scorers, target = builtin_suite("smoke")
    runner = EvalRunner(target=target, scorers=scorers)
    report = await runner.run(dataset)
    assert report.failed == 0


def test_builtin_suite_unknown_raises() -> None:
    with pytest.raises(KeyError):
        builtin_suite("does-not-exist")
