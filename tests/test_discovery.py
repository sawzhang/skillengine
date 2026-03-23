"""Tests for Agent Discovery."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from skillkit.a2a.agent_card import AgentCard, AgentCardSkill
from skillkit.a2a.discovery import (
    AGENT_DISCOVERED,
    AGENT_HEALTH_CHANGED,
    AGENT_REMOVED,
    AgentDiscovery,
    AgentHealthStatus,
    DiscoveryConfig,
)
from skillkit.a2a.registry import AgentRegistry
from skillkit.models import Skill, SkillMetadata


def _make_card(name: str, description: str = "Test agent") -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        skills=[AgentCardSkill(name=name, description=description)],
    )


class TestDiscoveryEndpoint:
    @pytest.mark.asyncio
    async def test_discover_new_agents(self):
        registry = AgentRegistry()
        client = MagicMock()
        client.discover_all = AsyncMock(
            return_value=[_make_card("agent-a"), _make_card("agent-b")]
        )

        discovery = AgentDiscovery(registry=registry, client=client)
        new = await discovery.discover_endpoint("http://remote:8080")

        assert len(new) == 2
        assert "agent-a" in new
        assert "agent-b" in new
        assert registry.count == 2

    @pytest.mark.asyncio
    async def test_discover_existing_agent_no_duplicate(self):
        registry = AgentRegistry()
        card = _make_card("agent-a")
        registry.register_remote(card, "http://remote:8080")

        client = MagicMock()
        client.discover_all = AsyncMock(return_value=[_make_card("agent-a")])

        discovery = AgentDiscovery(registry=registry, client=client)
        new = await discovery.discover_endpoint("http://remote:8080")

        assert len(new) == 0  # Not new
        assert registry.count == 1  # No duplicate

    @pytest.mark.asyncio
    async def test_discover_updates_version(self):
        registry = AgentRegistry()
        old_card = _make_card("agent-a")
        old_card.version = "1.0.0"
        registry.register_remote(old_card, "http://remote:8080")

        new_card = _make_card("agent-a")
        new_card.version = "2.0.0"

        client = MagicMock()
        client.discover_all = AsyncMock(return_value=[new_card])

        discovery = AgentDiscovery(registry=registry, client=client)
        await discovery.discover_endpoint("http://remote:8080")

        assert registry.get("agent-a").card.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_discover_failure_graceful(self):
        registry = AgentRegistry()
        client = MagicMock()
        client.discover_all = AsyncMock(side_effect=Exception("connection refused"))

        discovery = AgentDiscovery(registry=registry, client=client)
        new = await discovery.discover_endpoint("http://down:8080")

        assert len(new) == 0

    @pytest.mark.asyncio
    async def test_discover_removes_vanished_agents(self):
        registry = AgentRegistry()
        client = MagicMock()

        # First discovery: 2 agents
        client.discover_all = AsyncMock(
            return_value=[_make_card("a"), _make_card("b")]
        )
        discovery = AgentDiscovery(registry=registry, client=client)
        await discovery.discover_endpoint("http://remote:8080")
        assert registry.count == 2

        # Second discovery: only agent-a remains
        client.discover_all = AsyncMock(return_value=[_make_card("a")])
        await discovery.discover_endpoint("http://remote:8080")

        assert registry.count == 1
        assert registry.get("a") is not None
        assert registry.get("b") is None

    @pytest.mark.asyncio
    async def test_discover_emits_events(self):
        registry = AgentRegistry()
        event_bus = MagicMock()
        event_bus.emit = AsyncMock()

        client = MagicMock()
        client.discover_all = AsyncMock(return_value=[_make_card("new-agent")])

        discovery = AgentDiscovery(
            registry=registry, client=client, event_bus=event_bus
        )
        await discovery.discover_endpoint("http://remote:8080")

        event_bus.emit.assert_called()
        call_args = event_bus.emit.call_args_list
        assert any(c[0][0] == AGENT_DISCOVERED for c in call_args)


class TestDiscoverAll:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        registry = AgentRegistry()
        client = MagicMock()
        client.discover_all = AsyncMock(return_value=[_make_card("a")])

        config = DiscoveryConfig(
            endpoints=["http://ep1:8080", "http://ep2:8080"]
        )
        discovery = AgentDiscovery(
            registry=registry, config=config, client=client
        )
        event = await discovery.discover_all()

        assert event.endpoints_checked == 2
        assert client.discover_all.call_count == 2


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_agent(self):
        registry = AgentRegistry()
        card = _make_card("healthy-agent")
        registry.register_remote(card, "http://remote:8080")

        client = MagicMock()
        client.discover = AsyncMock(return_value=card)

        discovery = AgentDiscovery(registry=registry, client=client)
        results = await discovery.check_health()

        assert results["healthy-agent"] == AgentHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_after_failures(self):
        registry = AgentRegistry()
        card = _make_card("flaky-agent")
        registry.register_remote(card, "http://remote:8080")

        client = MagicMock()
        client.discover = AsyncMock(side_effect=Exception("timeout"))

        discovery = AgentDiscovery(registry=registry, client=client)

        # First failure
        results = await discovery.check_health()
        assert results["flaky-agent"] == AgentHealthStatus.UNKNOWN  # Not enough failures yet

        # Second failure → UNHEALTHY
        results = await discovery.check_health()
        assert results["flaky-agent"] == AgentHealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_auto_remove_unhealthy(self):
        registry = AgentRegistry()
        card = _make_card("doomed-agent")
        registry.register_remote(card, "http://remote:8080")

        client = MagicMock()
        client.discover = AsyncMock(side_effect=Exception("gone"))

        config = DiscoveryConfig(
            auto_remove_unhealthy=True,
            max_consecutive_failures=2,
        )
        discovery = AgentDiscovery(
            registry=registry, config=config, client=client
        )

        await discovery.check_health()  # failure 1
        assert registry.count == 1

        await discovery.check_health()  # failure 2 → remove
        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_recovery_from_unhealthy(self):
        registry = AgentRegistry()
        card = _make_card("recovering")
        registry.register_remote(card, "http://remote:8080")

        client = MagicMock()

        discovery = AgentDiscovery(registry=registry, client=client)

        # Make unhealthy
        client.discover = AsyncMock(side_effect=Exception("down"))
        await discovery.check_health()
        await discovery.check_health()
        assert discovery.get_health("recovering") == AgentHealthStatus.UNHEALTHY

        # Recover
        client.discover = AsyncMock(return_value=card)
        await discovery.check_health()
        assert discovery.get_health("recovering") == AgentHealthStatus.HEALTHY

    def test_get_health_unknown(self):
        discovery = AgentDiscovery(registry=AgentRegistry())
        assert discovery.get_health("nonexistent") == AgentHealthStatus.UNKNOWN

    def test_health_report(self):
        discovery = AgentDiscovery(registry=AgentRegistry())
        report = discovery.health_report()
        assert isinstance(report, dict)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        registry = AgentRegistry()
        client = MagicMock()
        client.discover_all = AsyncMock(return_value=[])
        client.discover = AsyncMock(return_value=_make_card("x"))

        config = DiscoveryConfig(
            endpoints=["http://ep:8080"],
            refresh_interval_seconds=100,
            health_check_interval_seconds=100,
        )
        discovery = AgentDiscovery(
            registry=registry, config=config, client=client
        )

        await discovery.start()
        assert discovery.is_running

        await discovery.stop()
        assert not discovery.is_running
