"""Tests for CTX-1: SummarizingCompactor and ToolResultTruncator."""

from __future__ import annotations

import pytest

from skillengine import (
    SummarizingCompactor,
    TokenBudgetCompactor,
    ToolResultTruncator,
    estimate_messages_tokens,
)
from skillengine.agent import AgentMessage
from skillengine.models import TextContent


def _msg(role: str, content, **kw) -> AgentMessage:
    return AgentMessage(role=role, content=content, **kw)


# ---------------------------------------------------------------------------
# ToolResultTruncator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_result_truncator_shortens_old_results() -> None:
    big = "x" * 20_000
    msgs = [
        _msg("user", "do thing"),
        _msg("assistant", "calling tool"),
        _msg("tool", big, name="search", tool_call_id="1"),
        _msg("assistant", "calling tool"),
        _msg("tool", "small", name="search", tool_call_id="2"),
    ]
    t = ToolResultTruncator(max_chars=200, keep_recent=1)
    out = await t.compact(msgs, budget_tokens=10_000)
    # last tool result preserved
    assert out[-1].content == "small"
    # older tool result truncated
    assert len(out[2].content) <= 200
    assert "truncated" in out[2].content


@pytest.mark.asyncio
async def test_tool_result_truncator_keeps_recent_intact() -> None:
    big = "y" * 5000
    msgs = [
        _msg("tool", big, name="a", tool_call_id="1"),
        _msg("tool", big, name="b", tool_call_id="2"),
        _msg("tool", big, name="c", tool_call_id="3"),
    ]
    t = ToolResultTruncator(max_chars=200, keep_recent=2)
    out = await t.compact(msgs, budget_tokens=10_000)
    assert len(out[0].content) <= 200  # oldest truncated
    assert out[1].content == big  # recent kept
    assert out[2].content == big


@pytest.mark.asyncio
async def test_tool_result_truncator_handles_text_blocks() -> None:
    big = "z" * 8000
    msgs = [
        _msg("tool", [TextContent(text=big)], name="search", tool_call_id="1"),
        _msg("tool", "tail", name="search", tool_call_id="2"),
    ]
    t = ToolResultTruncator(max_chars=300, keep_recent=1)
    out = await t.compact(msgs, budget_tokens=10_000)
    assert isinstance(out[0].content, list)
    assert len(out[0].content[0].text) <= 300


@pytest.mark.asyncio
async def test_tool_result_truncator_does_not_modify_non_tool_messages() -> None:
    big = "a" * 9000
    msgs = [
        _msg("user", big),
        _msg("assistant", big),
        _msg("tool", big, name="x", tool_call_id="1"),
    ]
    t = ToolResultTruncator(max_chars=100, keep_recent=0)
    out = await t.compact(msgs, budget_tokens=10_000)
    assert out[0].content == big
    assert out[1].content == big
    assert len(out[2].content) <= 100


@pytest.mark.asyncio
async def test_tool_result_truncator_no_tool_messages() -> None:
    msgs = [_msg("user", "hi"), _msg("assistant", "yo")]
    t = ToolResultTruncator(max_chars=100)
    out = await t.compact(msgs, budget_tokens=1000)
    assert [m.content for m in out] == ["hi", "yo"]


def test_tool_result_truncator_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        ToolResultTruncator(max_chars=10)
    with pytest.raises(ValueError):
        ToolResultTruncator(max_chars=200, keep_recent=-1)


# ---------------------------------------------------------------------------
# SummarizingCompactor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarizing_compactor_calls_summarizer_and_keeps_tail() -> None:
    seen: list[int] = []

    def summ(msgs: list[AgentMessage]) -> str:
        seen.append(len(msgs))
        return "SUMMARY"

    msgs = [_msg("user", "x" * 2000) for _ in range(10)]
    c = SummarizingCompactor(summarizer=summ, keep_recent=3)
    out = await c.compact(msgs, budget_tokens=2000)
    # First message is the synthesized summary, then 3 recent.
    assert out[0].role == "system"
    assert "SUMMARY" in out[0].content
    assert out[0].metadata.get("compacted") is True
    assert out[0].metadata.get("compacted_messages") == 7
    assert len(out) == 4
    assert seen == [7]


@pytest.mark.asyncio
async def test_summarizing_compactor_supports_async_summarizer() -> None:
    async def summ(msgs: list[AgentMessage]) -> str:
        return f"summary-of-{len(msgs)}"

    msgs = [_msg("user", "x" * 2000) for _ in range(6)]
    c = SummarizingCompactor(summarizer=summ, keep_recent=2)
    out = await c.compact(msgs, budget_tokens=100)
    assert "summary-of-4" in out[0].content


@pytest.mark.asyncio
async def test_summarizing_compactor_under_budget_passes_through() -> None:
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    c = SummarizingCompactor(summarizer=lambda _: "S", keep_recent=1)
    out = await c.compact(msgs, budget_tokens=10_000)
    assert [m.content for m in out] == ["hello", "hi"]


@pytest.mark.asyncio
async def test_summarizing_compactor_falls_back_if_summarizer_raises() -> None:
    def boom(msgs: list[AgentMessage]) -> str:
        raise RuntimeError("nope")

    msgs = [_msg("user", "x" * 4000) for _ in range(8)]
    c = SummarizingCompactor(summarizer=boom, keep_recent=2)
    out = await c.compact(msgs, budget_tokens=100)
    # Should still produce a summary message with an error note.
    assert out[0].role == "system"
    assert "summary unavailable" in out[0].content


@pytest.mark.asyncio
async def test_summarizing_compactor_default_summarizer_works() -> None:
    msgs = [_msg("user", f"message {i}") for i in range(10)]
    c = SummarizingCompactor(keep_recent=2)  # uses default summarizer
    out = await c.compact(msgs, budget_tokens=50)
    assert out[0].role == "system"
    assert "USER" in out[0].content  # default summarizer includes role


@pytest.mark.asyncio
async def test_summarizing_compactor_short_history_kept_intact() -> None:
    msgs = [_msg("user", "a"), _msg("assistant", "b")]
    c = SummarizingCompactor(summarizer=lambda _: "S", keep_recent=5)
    out = await c.compact(msgs, budget_tokens=1)  # very tight budget
    # Fewer messages than keep_recent → no summarization performed.
    assert len(out) == 2


def test_summarizing_compactor_rejects_invalid_keep_recent() -> None:
    with pytest.raises(ValueError):
        SummarizingCompactor(keep_recent=0)


@pytest.mark.asyncio
async def test_summarizing_compactor_fallback_when_tail_still_too_big() -> None:
    # tail tokens >> budget → falls back to TokenBudgetCompactor on result.
    msgs = [_msg("user", "x" * 8000) for _ in range(8)]
    c = SummarizingCompactor(summarizer=lambda _: "S", keep_recent=4)
    out = await c.compact(msgs, budget_tokens=200)
    # Should not crash and should not exceed budget by much.
    assert isinstance(out, list)
    # Fallback ensures we end up with at least the most recent message.
    assert out[-1].content == "x" * 8000


@pytest.mark.asyncio
async def test_tool_result_truncator_then_summarizer_pipeline() -> None:
    """Smoke test: a typical pipeline composition."""
    big = "q" * 6000
    msgs = [
        _msg("user", "task"),
        _msg("assistant", "calling tool"),
        _msg("tool", big, name="s", tool_call_id="1"),
        _msg("assistant", "result"),
        _msg("user", "next"),
        _msg("assistant", "calling tool again"),
        _msg("tool", big, name="s", tool_call_id="2"),
        _msg("assistant", "final"),
    ]
    before_tokens = estimate_messages_tokens(msgs)

    trunc = ToolResultTruncator(max_chars=400, keep_recent=1)
    msgs = await trunc.compact(msgs, budget_tokens=1000)

    summ = SummarizingCompactor(summarizer=lambda _: "history", keep_recent=3)
    msgs = await summ.compact(msgs, budget_tokens=200)

    after_tokens = estimate_messages_tokens(msgs)
    assert after_tokens < before_tokens
    assert msgs[0].role == "system"
    # Verify TokenBudgetCompactor would also accept this (sanity).
    fb = TokenBudgetCompactor()
    final = await fb.compact(msgs, budget_tokens=10_000)
    assert len(final) >= 1
