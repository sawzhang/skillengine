"""
A2A (Agent-to-Agent) integration for SkillEngine.

Provides Agent Card generation, agent registry, and A2A protocol support.
Skills are the single source of truth — Agent Cards are derived automatically.
"""

from skillengine.a2a.agent_card import AgentCapabilities, AgentCard, AgentCardSkill
from skillengine.a2a.client import A2AClient, create_remote_agent_tool
from skillengine.a2a.coordinator import CoordinatorAgent, CoordinatorConfig
from skillengine.a2a.discovery import (
    AGENT_DISCOVERED,
    AGENT_HEALTH_CHANGED,
    AGENT_REMOVED,
    DISCOVERY_CYCLE_COMPLETE,
    AgentDiscovery,
    AgentHealthStatus,
    DiscoveryConfig,
)
from skillengine.a2a.handoffs import (
    Handoff,
    HandoffTarget,
    InputFilter,
    a2a_handoff,
    agent_handoff,
    callable_handoff,
    handoff,
    to_tool_definition,
)
from skillengine.a2a.models import A2ATaskRequest, A2ATaskResponse, TaskStatus
from skillengine.a2a.registry import AgentRegistry, AgentSource, AgentStats, RegisteredAgent
from skillengine.a2a.router import PerformanceRouter, RouteResult, RoutingConfig

__all__ = [
    # Agent Card
    "AgentCard",
    "AgentCardSkill",
    "AgentCapabilities",
    # Registry
    "AgentRegistry",
    "RegisteredAgent",
    "AgentSource",
    "AgentStats",
    # Models
    "A2ATaskRequest",
    "A2ATaskResponse",
    "TaskStatus",
    # Client
    "A2AClient",
    "create_remote_agent_tool",
    # Discovery
    "AgentDiscovery",
    "DiscoveryConfig",
    "AgentHealthStatus",
    "AGENT_DISCOVERED",
    "AGENT_REMOVED",
    "AGENT_HEALTH_CHANGED",
    "DISCOVERY_CYCLE_COMPLETE",
    # Router
    "PerformanceRouter",
    "RoutingConfig",
    "RouteResult",
    # Coordinator
    "CoordinatorAgent",
    "CoordinatorConfig",
    # Handoffs (OpenAI Agents SDK / Anthropic A2A draft compat shim)
    "Handoff",
    "HandoffTarget",
    "InputFilter",
    "handoff",
    "callable_handoff",
    "agent_handoff",
    "a2a_handoff",
    "to_tool_definition",
]
