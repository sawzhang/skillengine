"""Cron-based prompt scheduler.

Maps standard 5-field cron expressions ("minute hour day-of-month month day-of-week")
to prompts that get injected back into an :class:`AgentRunner` via ``follow_up()``.

Example::

    scheduler = CronScheduler()
    await scheduler.start(callback=lambda job: agent.follow_up(job.prompt))
    scheduler.add("*/5 * * * *", "Check the deploy status")
    ...
    await scheduler.stop()

Scope (v0):
- In-memory only; jobs are lost on REPL exit.
- Pure stdlib parser; numeric fields only (no Mon/Jan aliases).
- Local timezone (whatever ``datetime.now()`` reports).
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from skillengine.logging import get_logger

logger = get_logger("scheduler")

# Field bounds: (low, high) inclusive
_FIELD_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),  # day of week (0=Sunday)
)


def _parse_field(token: str, lo: int, hi: int) -> frozenset[int]:
    """Parse one cron field. Supports ``*``, ``*/N``, ``a-b``, ``a,b,c``."""
    if not token:
        raise ValueError("empty field")

    out: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty list item in {token!r}")

        step = 1
        if "/" in part:
            base, step_s = part.split("/", 1)
            try:
                step = int(step_s)
            except ValueError as e:
                raise ValueError(f"invalid step in {part!r}") from e
            if step <= 0:
                raise ValueError(f"step must be positive in {part!r}")
        else:
            base = part

        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            a_s, b_s = base.split("-", 1)
            try:
                start, end = int(a_s), int(b_s)
            except ValueError as e:
                raise ValueError(f"invalid range in {part!r}") from e
            if start > end:
                raise ValueError(f"reversed range in {part!r}")
        else:
            try:
                start = end = int(base)
            except ValueError as e:
                raise ValueError(f"invalid value in {part!r}") from e

        if start < lo or end > hi:
            raise ValueError(f"out of bounds {part!r}; expected [{lo},{hi}]")

        out.update(range(start, end + 1, step))

    if not out:
        raise ValueError(f"field {token!r} matches nothing")
    return frozenset(out)


@dataclass(frozen=True)
class CronExpression:
    """A parsed 5-field cron expression."""

    raw: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]

    @classmethod
    def parse(cls, expr: str) -> CronExpression:
        tokens = expr.split()
        if len(tokens) != 5:
            raise ValueError(
                f"expected 5 fields (minute hour day month dow), got {len(tokens)}: {expr!r}"
            )
        fields = tuple(_parse_field(tok, lo, hi) for tok, (lo, hi) in zip(tokens, _FIELD_BOUNDS))
        return cls(
            raw=expr,
            minutes=fields[0],
            hours=fields[1],
            days=fields[2],
            months=fields[3],
            weekdays=fields[4],
        )

    def matches(self, dt: datetime) -> bool:
        # cron weekday: 0=Sunday..6=Saturday; Python: Monday=0..Sunday=6
        cron_dow = (dt.weekday() + 1) % 7
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.day in self.days
            and dt.month in self.months
            and cron_dow in self.weekdays
        )

    def next_after(self, now: datetime) -> datetime:
        """Return the next datetime strictly greater than ``now`` that matches.

        Truncates seconds/microseconds. Linear minute-by-minute scan capped at
        ~2 years to detect impossible expressions (e.g., ``* * 31 2 *``).
        """
        candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        # 2 years cap: catches Feb 31 etc., still cheap (~1M iterations max).
        deadline = candidate + timedelta(days=366 * 2)
        while candidate < deadline:
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(f"cron expression {self.raw!r} has no match within 2 years")


@dataclass
class CronJob:
    """A scheduled prompt-injection job."""

    id: str
    cron: str
    expression: CronExpression
    prompt: str
    recurring: bool
    created_at: datetime
    next_fire_at: datetime
    last_fired_at: datetime | None = None
    fire_count: int = 0


CronCallback = Callable[[CronJob], Awaitable[None]]


class CronScheduler:
    """Manages a set of cron jobs and fires their callbacks when due.

    Single background asyncio task sleeps until the earliest ``next_fire_at``,
    then invokes the callback. ``add()`` and ``delete()`` wake the loop so it
    re-evaluates the next deadline immediately.
    """

    def __init__(self, *, clock: Callable[[], datetime] = datetime.now) -> None:
        self._jobs: dict[str, CronJob] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._callback: CronCallback | None = None
        self._clock = clock
        self._stopping = False

    # ── Public API ───────────────────────────────────────────────────

    def add(self, cron: str, prompt: str, *, recurring: bool = True) -> CronJob:
        """Add a new cron job. Returns the created :class:`CronJob`.

        Raises:
            ValueError: invalid cron expression or empty prompt.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        expression = CronExpression.parse(cron)
        now = self._clock()
        job = CronJob(
            id=secrets.token_hex(4),
            cron=cron,
            expression=expression,
            prompt=prompt,
            recurring=recurring,
            created_at=now,
            next_fire_at=expression.next_after(now),
        )
        self._jobs[job.id] = job
        self._wake.set()
        return job

    def delete(self, job_id: str) -> bool:
        """Remove a job by id. Returns True if removed, False if not found."""
        removed = self._jobs.pop(job_id, None) is not None
        if removed:
            self._wake.set()
        return removed

    def list(self) -> list[CronJob]:
        """Return a snapshot of jobs ordered by next fire time."""
        return sorted(self._jobs.values(), key=lambda j: j.next_fire_at)

    def get(self, job_id: str) -> CronJob | None:
        return self._jobs.get(job_id)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, callback: CronCallback) -> None:
        """Start the background scheduler loop.

        ``callback`` is invoked once per fire with the job. Exceptions raised
        by the callback are logged and the scheduler keeps running.
        """
        if self.is_running:
            return
        self._callback = callback
        self._stopping = False
        self._task = asyncio.create_task(self._run_loop(), name="cron-scheduler")

    async def stop(self) -> None:
        """Stop the scheduler. Pending jobs are kept (in case caller restarts)."""
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ── Internal loop ────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while not self._stopping:
            try:
                if not self._jobs:
                    self._wake.clear()
                    await self._wake.wait()
                    continue

                job = min(self._jobs.values(), key=lambda j: j.next_fire_at)
                now = self._clock()
                delay = (job.next_fire_at - now).total_seconds()

                if delay > 0:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=delay)
                        # Mutation woke us — recompute next deadline.
                        continue
                    except asyncio.TimeoutError:
                        pass  # deadline elapsed, fall through to fire

                # Re-check job is still present (could have been deleted while sleeping)
                if job.id not in self._jobs:
                    continue

                await self._fire(job)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")
                # Avoid tight error loop
                await asyncio.sleep(1.0)

    async def _fire(self, job: CronJob) -> None:
        now = self._clock()
        job.last_fired_at = now
        job.fire_count += 1

        if job.recurring:
            try:
                job.next_fire_at = job.expression.next_after(now)
            except ValueError:
                logger.exception("Failed to compute next_fire for %s; removing", job.id)
                self._jobs.pop(job.id, None)
        else:
            self._jobs.pop(job.id, None)

        if self._callback is None:
            return
        try:
            await self._callback(job)
        except Exception:
            logger.exception("Cron callback raised for job %s", job.id)
