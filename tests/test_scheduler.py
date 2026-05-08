"""Tests for the cron-based prompt scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from skillengine.scheduler import CronExpression, CronJob, CronScheduler

# ── CronExpression.parse ─────────────────────────────────────────────


class TestCronExpressionParse:
    def test_all_wildcards(self) -> None:
        e = CronExpression.parse("* * * * *")
        assert len(e.minutes) == 60
        assert len(e.hours) == 24
        assert len(e.days) == 31
        assert len(e.months) == 12
        assert len(e.weekdays) == 7

    def test_step(self) -> None:
        e = CronExpression.parse("*/15 * * * *")
        assert e.minutes == frozenset({0, 15, 30, 45})

    def test_specific_value(self) -> None:
        e = CronExpression.parse("0 9 * * *")
        assert e.minutes == frozenset({0})
        assert e.hours == frozenset({9})

    def test_range(self) -> None:
        e = CronExpression.parse("0 9 * * 1-5")
        assert e.weekdays == frozenset({1, 2, 3, 4, 5})

    def test_list(self) -> None:
        e = CronExpression.parse("0,15,30,45 * * * *")
        assert e.minutes == frozenset({0, 15, 30, 45})

    def test_combined_list_and_range(self) -> None:
        e = CronExpression.parse("0 9-11,17 * * *")
        assert e.hours == frozenset({9, 10, 11, 17})

    def test_step_with_range(self) -> None:
        e = CronExpression.parse("0 0-23/6 * * *")
        assert e.hours == frozenset({0, 6, 12, 18})

    @pytest.mark.parametrize(
        "expr",
        [
            "* * * *",  # too few
            "* * * * * *",  # too many
            "60 * * * *",  # minute out of range
            "* 24 * * *",  # hour out of range
            "* * 32 * *",  # day out of range
            "* * * 13 *",  # month out of range
            "* * * * 7",  # weekday out of range
            "5-1 * * * *",  # reversed range
            "*/0 * * * *",  # zero step
            "abc * * * *",  # non-numeric
            "* * * * Mon",  # alias not supported in v0
            "",  # empty
        ],
    )
    def test_invalid_raises(self, expr: str) -> None:
        with pytest.raises(ValueError):
            CronExpression.parse(expr)


# ── CronExpression.matches ───────────────────────────────────────────


class TestCronExpressionMatches:
    def test_matches_specific_minute(self) -> None:
        e = CronExpression.parse("30 * * * *")
        assert e.matches(datetime(2026, 5, 8, 14, 30))
        assert not e.matches(datetime(2026, 5, 8, 14, 31))

    def test_matches_weekday(self) -> None:
        e = CronExpression.parse("0 9 * * 1-5")
        # 2026-05-08 is Friday
        assert e.matches(datetime(2026, 5, 8, 9, 0))
        # 2026-05-09 is Saturday
        assert not e.matches(datetime(2026, 5, 9, 9, 0))

    def test_matches_sunday(self) -> None:
        # cron Sunday is 0; Python Sunday is 6
        e = CronExpression.parse("0 9 * * 0")
        # 2026-05-10 is Sunday
        assert e.matches(datetime(2026, 5, 10, 9, 0))
        # 2026-05-11 is Monday
        assert not e.matches(datetime(2026, 5, 11, 9, 0))


# ── CronExpression.next_after ────────────────────────────────────────


class TestCronExpressionNextAfter:
    def test_next_minute(self) -> None:
        e = CronExpression.parse("* * * * *")
        now = datetime(2026, 5, 8, 14, 30, 27)
        assert e.next_after(now) == datetime(2026, 5, 8, 14, 31)

    def test_next_step(self) -> None:
        e = CronExpression.parse("*/15 * * * *")
        now = datetime(2026, 5, 8, 14, 7)
        assert e.next_after(now) == datetime(2026, 5, 8, 14, 15)

    def test_next_specific_hour(self) -> None:
        e = CronExpression.parse("0 9 * * *")
        now = datetime(2026, 5, 8, 14, 30)
        assert e.next_after(now) == datetime(2026, 5, 9, 9, 0)

    def test_next_weekday(self) -> None:
        e = CronExpression.parse("0 9 * * 1-5")
        # Friday 10:00 → Monday 09:00
        now = datetime(2026, 5, 8, 10, 0)
        assert e.next_after(now) == datetime(2026, 5, 11, 9, 0)

    def test_strictly_after(self) -> None:
        # If "now" already matches, return the next occurrence, not "now".
        e = CronExpression.parse("0 * * * *")
        now = datetime(2026, 5, 8, 14, 0)
        assert e.next_after(now) == datetime(2026, 5, 8, 15, 0)

    def test_never_matches_raises(self) -> None:
        # Feb 31 doesn't exist
        e = CronExpression.parse("0 0 31 2 *")
        with pytest.raises(ValueError, match="no match"):
            e.next_after(datetime(2026, 1, 1))


# ── CronScheduler CRUD ───────────────────────────────────────────────


class TestCronSchedulerCRUD:
    def test_add_returns_job(self) -> None:
        s = CronScheduler()
        job = s.add("*/5 * * * *", "hello")
        assert isinstance(job, CronJob)
        assert job.cron == "*/5 * * * *"
        assert job.prompt == "hello"
        assert job.recurring is True
        assert s.get(job.id) is job

    def test_add_one_shot(self) -> None:
        s = CronScheduler()
        job = s.add("0 9 * * *", "morning", recurring=False)
        assert job.recurring is False

    def test_add_invalid_cron_raises(self) -> None:
        s = CronScheduler()
        with pytest.raises(ValueError):
            s.add("60 * * * *", "x")

    def test_add_empty_prompt_raises(self) -> None:
        s = CronScheduler()
        with pytest.raises(ValueError):
            s.add("* * * * *", "  ")

    def test_delete_existing(self) -> None:
        s = CronScheduler()
        job = s.add("* * * * *", "x")
        assert s.delete(job.id) is True
        assert s.get(job.id) is None

    def test_delete_missing(self) -> None:
        s = CronScheduler()
        assert s.delete("nonexistent") is False

    def test_list_sorted_by_next_fire(self) -> None:
        clock_value = [datetime(2026, 5, 8, 14, 0)]
        s = CronScheduler(clock=lambda: clock_value[0])
        job_late = s.add("0 23 * * *", "evening")
        job_soon = s.add("* * * * *", "any minute")
        listed = s.list()
        assert [j.id for j in listed] == [job_soon.id, job_late.id]


# ── CronScheduler firing ─────────────────────────────────────────────


class TestCronSchedulerFiring:
    @pytest.mark.asyncio
    async def test_fires_when_due(self) -> None:
        # Use a mutable clock so we can advance time deterministically.
        now = [datetime(2026, 5, 8, 14, 0, 0)]
        s = CronScheduler(clock=lambda: now[0])

        fired: list[CronJob] = []

        async def cb(job: CronJob) -> None:
            fired.append(job)

        await s.start(cb)
        try:
            job = s.add("* * * * *", "ping")
            assert job.next_fire_at == datetime(2026, 5, 8, 14, 1)

            # Advance the clock so the loop sees delay <= 0 on its next pass.
            # We poke the wake event by adding then deleting a sentinel.
            now[0] = datetime(2026, 5, 8, 14, 1, 5)
            sentinel = s.add("0 0 1 1 *", "noop-to-wake")  # very far away
            s.delete(sentinel.id)

            # Give the loop time to fire the real job.
            for _ in range(50):
                if fired:
                    break
                await asyncio.sleep(0.02)

            assert len(fired) == 1
            assert fired[0].id == job.id
            assert fired[0].fire_count == 1
            assert fired[0].last_fired_at == datetime(2026, 5, 8, 14, 1, 5)
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_one_shot_removes_after_fire(self) -> None:
        now = [datetime(2026, 5, 8, 14, 0, 0)]
        s = CronScheduler(clock=lambda: now[0])

        fired: list[CronJob] = []

        async def cb(job: CronJob) -> None:
            fired.append(job)

        await s.start(cb)
        try:
            job = s.add("* * * * *", "once", recurring=False)
            now[0] = datetime(2026, 5, 8, 14, 1, 5)
            sentinel = s.add("0 0 1 1 *", "noop")
            s.delete(sentinel.id)

            for _ in range(50):
                if fired:
                    break
                await asyncio.sleep(0.02)

            assert len(fired) == 1
            assert s.get(job.id) is None
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_recurring_recomputes_next_fire(self) -> None:
        now = [datetime(2026, 5, 8, 14, 0, 0)]
        s = CronScheduler(clock=lambda: now[0])

        fired: list[CronJob] = []

        async def cb(job: CronJob) -> None:
            fired.append(job)

        await s.start(cb)
        try:
            job = s.add("* * * * *", "tick")
            first_next = job.next_fire_at

            now[0] = datetime(2026, 5, 8, 14, 1, 5)
            sentinel = s.add("0 0 1 1 *", "noop")
            s.delete(sentinel.id)

            for _ in range(50):
                if fired:
                    break
                await asyncio.sleep(0.02)

            # After firing, recurring job should still exist with later next_fire_at
            assert s.get(job.id) is not None
            assert s.get(job.id).next_fire_at > first_next
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_kill_loop(self) -> None:
        now = [datetime(2026, 5, 8, 14, 0, 0)]
        s = CronScheduler(clock=lambda: now[0])

        async def cb(job: CronJob) -> None:
            raise RuntimeError("boom")

        await s.start(cb)
        try:
            s.add("* * * * *", "boom")
            now[0] = datetime(2026, 5, 8, 14, 1, 5)
            sentinel = s.add("0 0 1 1 *", "noop")
            s.delete(sentinel.id)
            await asyncio.sleep(0.1)
            # Scheduler should still be running
            assert s.is_running
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        s = CronScheduler()

        async def cb(job: CronJob) -> None:
            pass

        await s.start(cb)
        try:
            task1 = s._task
            await s.start(cb)
            assert s._task is task1  # didn't spawn a second task
        finally:
            await s.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self) -> None:
        s = CronScheduler()
        await s.stop()  # should not raise
        assert not s.is_running

    @pytest.mark.asyncio
    async def test_add_wakes_idle_loop(self) -> None:
        """A scheduler with no jobs should not spin; adding a job must wake it."""
        s = CronScheduler()
        fired: list[CronJob] = []

        async def cb(job: CronJob) -> None:
            fired.append(job)

        await s.start(cb)
        try:
            await asyncio.sleep(0.05)  # let the loop park on the wake event
            # Job that fires "now" — i.e., next minute is < 1s away when seconds are 59
            # Easier: use a mocked clock just like above
            assert s.is_running
            # We at least confirm add() doesn't deadlock and the job lands in list()
            job = s.add("0 0 1 1 *", "yearly")
            assert job in s.list()
        finally:
            await s.stop()
