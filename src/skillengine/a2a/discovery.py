"""Agent Discovery — automatic discovery and lifecycle management of A2A agents.

Provides:
1. Periodic discovery: refresh remote agent registrations on a schedule
2. Health checking: detect and mark unreachable agents
3. Event-driven updates: emit events on agent join/leave/health change

Integrates with TaskScheduler for scheduling and EventBus for notifications.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from skillengine.a2a.agent_card import AgentCard
from skillengine.a2a.client import A2AClient
from skillengine.a2a.registry import AgentRegistry

if TYPE_CHECKING:
    from skillengine.events import EventBus

logger = logging.getLogger(__name__)

# ── Event names ──────────────────────────────────────────────────────

AGENT_DISCOVERED = "agent_discovered"
AGENT_REMOVED = "agent_removed"
AGENT_HEALTH_CHANGED = "agent_health_changed"
DISCOVERY_CYCLE_COMPLETE = "discovery_cycle_complete"


# ── Event data ───────────────────────────────────────────────────────


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentDiscoveredEvent:
    """Emitted when a new remote agent is discovered."""

    agent_name: str
    endpoint: str
    card: AgentCard


@dataclass
class AgentRemovedEvent:
    """Emitted when a previously known agent is no longer reachable."""

    agent_name: str
    endpoint: str
    reason: str


@dataclass
class AgentHealthChangedEvent:
    """Emitted when an agent's health status changes."""

    agent_name: str
    endpoint: str
    old_status: AgentHealthStatus
    new_status: AgentHealthStatus


@dataclass
class DiscoveryCycleCompleteEvent:
    """Emitted after a full discovery cycle."""

    endpoints_checked: int
    agents_discovered: int
    agents_removed: int
    healthy: int
    unhealthy: int
    duration_ms: float


# ── Discovery config ─────────────────────────────────────────────────


@dataclass
class DiscoveryConfig:
    """Configuration for agent discovery."""

    # Endpoints to periodically discover
    endpoints: list[str] = field(default_factory=list)

    # Timing
    refresh_interval_seconds: float = 300.0  # 5 minutes
    health_check_interval_seconds: float = 60.0  # 1 minute
    request_timeout_seconds: float = 10.0

    # Behavior
    auto_remove_unhealthy: bool = False  # Remove agents after max_consecutive_failures
    max_consecutive_failures: int = 3
    remove_on_discovery_failure: bool = False  # Remove if endpoint stops responding


# ── Health tracking ──────────────────────────────────────────────────


@dataclass
class _AgentHealth:
    """Internal health state for a registered agent."""

    status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_check: float | None = None
    last_success: float | None = None
    last_error: str | None = None

    def record_success(self) -> AgentHealthStatus:
        old = self.status
        self.status = AgentHealthStatus.HEALTHY
        self.consecutive_failures = 0
        self.last_check = time.time()
        self.last_success = self.last_check
        self.last_error = None
        return old

    def record_failure(self, error: str) -> AgentHealthStatus:
        old = self.status
        self.consecutive_failures += 1
        self.last_check = time.time()
        self.last_error = error
        if self.consecutive_failures >= 2:
            self.status = AgentHealthStatus.UNHEALTHY
        return old


# ── Discovery Manager ────────────────────────────────────────────────


class AgentDiscovery:
    """Manages automatic discovery and health monitoring of remote A2A agents.

    Usage::

        discovery = AgentDiscovery(
            registry=registry,
            config=DiscoveryConfig(
                endpoints=["http://agent-a:8080", "http://agent-b:9090"],
                refresh_interval_seconds=300,
            ),
            event_bus=event_bus,
        )

        # Run one discovery cycle
        await discovery.discover_all()

        # Start periodic discovery (integrates with TaskScheduler)
        await discovery.start()

        # Or manually check health
        report = await discovery.check_health()
    """

    def __init__(
        self,
        registry: AgentRegistry,
        config: DiscoveryConfig | None = None,
        event_bus: EventBus | None = None,
        client: A2AClient | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or DiscoveryConfig()
        self.event_bus = event_bus
        self.client = client or A2AClient(timeout=self.config.request_timeout_seconds)

        # Internal state
        self._health: dict[str, _AgentHealth] = {}  # agent_name → health
        self._endpoint_agents: dict[str, set[str]] = {}  # endpoint → {agent_names}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ── Discovery ────────────────────────────────────────────────

    async def discover_endpoint(self, endpoint: str) -> list[str]:
        """Discover agents at a single endpoint.

        Returns list of newly registered agent names.
        """
        new_agents: list[str] = []
        try:
            cards = await self.client.discover_all(endpoint)
        except Exception as e:
            logger.warning("Discovery failed for %s: %s", endpoint, e)
            if self.config.remove_on_discovery_failure:
                removed = self._remove_endpoint_agents(endpoint, str(e))
                logger.info("Removed %d agents from unreachable %s", removed, endpoint)
            return []

        discovered_names: set[str] = set()
        for card in cards:
            discovered_names.add(card.name)
            existing = self.registry.get(card.name)

            if existing is None:
                # New agent
                self.registry.register_remote(card, endpoint=endpoint)
                self._health[card.name] = _AgentHealth(
                    status=AgentHealthStatus.HEALTHY,
                    last_check=time.time(),
                    last_success=time.time(),
                )
                new_agents.append(card.name)
                await self._emit(
                    AGENT_DISCOVERED,
                    AgentDiscoveredEvent(
                        agent_name=card.name,
                        endpoint=endpoint,
                        card=card,
                    ),
                )
                logger.info("Discovered new agent: %s at %s", card.name, endpoint)
            else:
                # Existing agent — update card if version changed
                if existing.card.version != card.version:
                    self.registry.register_remote(card, endpoint=endpoint)
                    logger.info(
                        "Updated agent %s: v%s → v%s",
                        card.name,
                        existing.card.version,
                        card.version,
                    )
                # Mark healthy on successful discovery
                health = self._health.setdefault(card.name, _AgentHealth())
                health.record_success()

        # Track which agents belong to this endpoint
        previous = self._endpoint_agents.get(endpoint, set())
        self._endpoint_agents[endpoint] = discovered_names

        # Remove agents that disappeared from this endpoint
        vanished = previous - discovered_names
        for name in vanished:
            self.registry.unregister(name)
            self._health.pop(name, None)
            await self._emit(
                AGENT_REMOVED,
                AgentRemovedEvent(
                    agent_name=name,
                    endpoint=endpoint,
                    reason="no longer advertised",
                ),
            )
            logger.info("Agent %s vanished from %s", name, endpoint)

        return new_agents

    async def discover_all(self) -> DiscoveryCycleCompleteEvent:
        """Run a full discovery cycle across all configured endpoints.

        Returns a summary event.
        """
        start = time.time()
        total_discovered = 0
        total_removed = 0

        for endpoint in self.config.endpoints:
            new = await self.discover_endpoint(endpoint)
            total_discovered += len(new)

        duration_ms = (time.time() - start) * 1000

        # Count health states
        healthy = sum(1 for h in self._health.values() if h.status == AgentHealthStatus.HEALTHY)
        unhealthy = sum(1 for h in self._health.values() if h.status == AgentHealthStatus.UNHEALTHY)

        event = DiscoveryCycleCompleteEvent(
            endpoints_checked=len(self.config.endpoints),
            agents_discovered=total_discovered,
            agents_removed=total_removed,
            healthy=healthy,
            unhealthy=unhealthy,
            duration_ms=duration_ms,
        )

        await self._emit(DISCOVERY_CYCLE_COMPLETE, event)
        return event

    # ── Health checking ──────────────────────────────────────────

    async def check_health(self) -> dict[str, AgentHealthStatus]:
        """Check health of all registered remote agents.

        Sends a lightweight request to each agent's endpoint.
        Returns mapping of agent_name → health status.
        """
        results: dict[str, AgentHealthStatus] = {}

        for agent in self.registry.remote_agents():
            if agent.endpoint is None:
                continue

            name = agent.card.name
            health = self._health.setdefault(name, _AgentHealth())

            try:
                # Use discover as a lightweight health check
                await self.client.discover(agent.endpoint)
                old_status = health.record_success()
                results[name] = AgentHealthStatus.HEALTHY

                if old_status != AgentHealthStatus.HEALTHY:
                    await self._emit(
                        AGENT_HEALTH_CHANGED,
                        AgentHealthChangedEvent(
                            agent_name=name,
                            endpoint=agent.endpoint,
                            old_status=old_status,
                            new_status=AgentHealthStatus.HEALTHY,
                        ),
                    )

            except Exception as e:
                old_status = health.record_failure(str(e))
                results[name] = health.status

                if old_status != health.status:
                    await self._emit(
                        AGENT_HEALTH_CHANGED,
                        AgentHealthChangedEvent(
                            agent_name=name,
                            endpoint=agent.endpoint,
                            old_status=old_status,
                            new_status=health.status,
                        ),
                    )

                # Auto-remove if configured
                if (
                    self.config.auto_remove_unhealthy
                    and health.consecutive_failures >= self.config.max_consecutive_failures
                ):
                    self.registry.unregister(name)
                    self._health.pop(name, None)
                    await self._emit(
                        AGENT_REMOVED,
                        AgentRemovedEvent(
                            agent_name=name,
                            endpoint=agent.endpoint,
                            reason=f"unhealthy after {health.consecutive_failures} failures",
                        ),
                    )
                    logger.warning(
                        "Removed unhealthy agent %s after %d failures",
                        name,
                        health.consecutive_failures,
                    )

        return results

    def get_health(self, agent_name: str) -> AgentHealthStatus:
        """Get the current health status of an agent."""
        health = self._health.get(agent_name)
        return health.status if health else AgentHealthStatus.UNKNOWN

    def health_report(self) -> dict[str, dict[str, Any]]:
        """Generate a full health report for all tracked agents."""
        report = {}
        for name, health in self._health.items():
            report[name] = {
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "last_check": health.last_check,
                "last_success": health.last_success,
                "last_error": health.last_error,
            }
        return report

    # ── Periodic lifecycle ───────────────────────────────────────

    async def start(self) -> None:
        """Start periodic discovery and health checking.

        Spawns background tasks for:
        1. Periodic discovery (refresh_interval_seconds)
        2. Periodic health checks (health_check_interval_seconds)
        """
        if self._running:
            return

        self._running = True

        # Initial discovery
        await self.discover_all()

        # Start periodic tasks
        if self.config.endpoints:
            self._tasks.append(
                asyncio.create_task(self._discovery_loop(), name="a2a-discovery")
            )
            self._tasks.append(
                asyncio.create_task(self._health_check_loop(), name="a2a-health")
            )

        logger.info(
            "Agent discovery started: %d endpoints, refresh=%ds, health=%ds",
            len(self.config.endpoints),
            self.config.refresh_interval_seconds,
            self.config.health_check_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop periodic discovery and health checking."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Agent discovery stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    async def _discovery_loop(self) -> None:
        """Background loop for periodic agent discovery."""
        while self._running:
            try:
                await asyncio.sleep(self.config.refresh_interval_seconds)
                if not self._running:
                    break
                await self.discover_all()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Discovery cycle error")

    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checking."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                if not self._running:
                    break
                await self.check_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health check error")

    # ── Helpers ──────────────────────────────────────────────────

    def _remove_endpoint_agents(self, endpoint: str, reason: str) -> int:
        """Remove all agents associated with an endpoint."""
        names = self._endpoint_agents.pop(endpoint, set())
        for name in names:
            self.registry.unregister(name)
            self._health.pop(name, None)
        return len(names)

    async def _emit(self, event_name: str, data: Any) -> None:
        """Emit an event if EventBus is available."""
        if self.event_bus is not None:
            try:
                await self.event_bus.emit(event_name, data)
            except Exception:
                logger.debug("Failed to emit event %s", event_name, exc_info=True)
