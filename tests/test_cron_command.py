"""Tests for the /cron slash command handler."""

from __future__ import annotations

import asyncio

import pytest

from skillengine.commands import CommandResult, make_cron_handler
from skillengine.scheduler import CronScheduler


@pytest.fixture
def scheduler() -> CronScheduler:
    return CronScheduler()


async def _dispatch(handler, args: str) -> CommandResult:
    result = handler(args)
    if asyncio.iscoroutine(result):
        result = await result
    return result


@pytest.mark.asyncio
async def test_list_empty(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "")
    assert "no cron jobs" in result.output


@pytest.mark.asyncio
async def test_add_creates_job(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, '''add "*/5 * * * *" "Check the deploy status"''')
    assert not result.error
    assert "Added recurring job" in result.output
    assert len(scheduler.list()) == 1
    job = scheduler.list()[0]
    assert job.cron == "*/5 * * * *"
    assert job.prompt == "Check the deploy status"
    assert job.recurring is True


@pytest.mark.asyncio
async def test_once_creates_one_shot(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, '''once "0 9 * * *" "morning brief"''')
    assert not result.error
    assert "Added one-shot job" in result.output
    assert scheduler.list()[0].recurring is False


@pytest.mark.asyncio
async def test_add_invalid_cron(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, '''add "60 * * * *" "x"''')
    assert result.error
    assert "Invalid cron job" in result.error
    assert scheduler.list() == []


@pytest.mark.asyncio
async def test_add_missing_args(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "add")
    assert result.error
    assert "Usage:" in result.error


@pytest.mark.asyncio
async def test_unbalanced_quotes(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, 'add "* * * * *" "missing close')
    assert result.error
    assert "parse" in result.error.lower()


@pytest.mark.asyncio
async def test_list_with_jobs(scheduler: CronScheduler) -> None:
    scheduler.add("*/5 * * * *", "first")
    scheduler.add("0 9 * * *", "second")
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "list")
    assert "first" in result.output
    assert "second" in result.output
    assert "*/5 * * * *" in result.output


@pytest.mark.asyncio
async def test_delete(scheduler: CronScheduler) -> None:
    job = scheduler.add("* * * * *", "x")
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, f"delete {job.id}")
    assert "Deleted" in result.output
    assert scheduler.list() == []


@pytest.mark.asyncio
async def test_delete_missing_id(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "delete nonexistent")
    assert "No job" in result.output


@pytest.mark.asyncio
async def test_delete_usage(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "delete")
    assert result.error
    assert "Usage" in result.error


@pytest.mark.asyncio
async def test_clear(scheduler: CronScheduler) -> None:
    scheduler.add("* * * * *", "a")
    scheduler.add("0 9 * * *", "b")
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "clear")
    assert "Cleared 2" in result.output
    assert scheduler.list() == []


@pytest.mark.asyncio
async def test_unknown_subcommand(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    result = await _dispatch(handler, "foobar")
    assert result.error
    assert "Unknown subcommand" in result.error


@pytest.mark.asyncio
async def test_prompt_with_spaces_preserved(scheduler: CronScheduler) -> None:
    handler = make_cron_handler(scheduler)
    await _dispatch(handler, 'add "* * * * *" "say hello to the world"')
    assert scheduler.list()[0].prompt == "say hello to the world"
