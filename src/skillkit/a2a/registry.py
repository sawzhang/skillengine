"""Agent Registry — unified registration, discovery, and routing of agents.

Manages two types of agents:
1. Local agents (derived from Skills, executed in-process)
2. Remote agents (discovered via A2A protocol, called over HTTP)

The registry provides:
- Agent Card indexing for Orchestrator awareness
- Keyword-based routing (Phase 1)
- Stats tracking for performance-driven routing (Phase 2+)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from skillkit.a2a.agent_card import AgentCard

if TYPE_CHECKING:
    from skillkit.models import Skill

logger = logging.getLogger(__name__)


class AgentSource(str, Enum):
    """Where an agent was registered from."""

    LOCAL = "local"
    REMOTE = "remote"


@dataclass
class AgentStats:
    """Agent performance metrics (dynamic understanding layer).

    Tracks call history to enable performance-driven routing:
    - High success rate → prefer this agent for similar tasks
    - Low latency → prefer for time-sensitive tasks
    """

    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_called: float | None = None
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def record_success(self, latency_ms: float) -> None:
        self.total_calls += 1
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_called = time.time()

    def record_failure(self, error: str, latency_ms: float) -> None:
        self.total_calls += 1
        self.failure_count += 1
        self.total_latency_ms += latency_ms
        self.last_called = time.time()
        self.last_error = error


@dataclass
class RegisteredAgent:
    """An agent registered in the registry."""

    card: AgentCard
    source: AgentSource = AgentSource.LOCAL
    skill: Skill | None = None  # Only for local agents
    endpoint: str | None = None  # Only for remote agents
    stats: AgentStats = field(default_factory=AgentStats)


class AgentRegistry:
    """Unified registry for local and remote agents.

    Core responsibilities:
    1. Register agents (from Skills or A2A discovery)
    2. Generate system prompt awareness (cards_summary)
    3. Route queries to matching agents (keyword → semantic)
    4. Track agent performance stats
    """

    def __init__(self) -> None:
        self._agents: dict[str, RegisteredAgent] = {}

    # ── Registration ──────────────────────────────────────────────

    def register_skill(
        self,
        skill: Skill,
        base_url: str | None = None,
    ) -> RegisteredAgent:
        """Register a local agent from a Skill definition.

        Args:
            skill: The Skill to register.
            base_url: Optional base URL if this agent is also exposed via A2A.

        Returns:
            The registered agent entry.
        """
        card = AgentCard.from_skill(skill, base_url)
        agent = RegisteredAgent(
            card=card,
            source=AgentSource.LOCAL,
            skill=skill,
        )
        self._agents[skill.name] = agent
        logger.debug("Registered local agent: %s", skill.name)
        return agent

    def register_remote(
        self,
        card: AgentCard,
        endpoint: str,
    ) -> RegisteredAgent:
        """Register a remote agent from an A2A Agent Card.

        Args:
            card: The Agent Card received from the remote agent.
            endpoint: The A2A endpoint URL.

        Returns:
            The registered agent entry.
        """
        agent = RegisteredAgent(
            card=card,
            source=AgentSource.REMOTE,
            endpoint=endpoint,
        )
        self._agents[card.name] = agent
        logger.debug("Registered remote agent: %s at %s", card.name, endpoint)
        return agent

    def unregister(self, name: str) -> bool:
        """Remove an agent from the registry."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    # ── Queries ───────────────────────────────────────────────────

    def get(self, name: str) -> RegisteredAgent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def all(self) -> list[RegisteredAgent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def local_agents(self) -> list[RegisteredAgent]:
        """Get all local (Skill-based) agents."""
        return [a for a in self._agents.values() if a.source == AgentSource.LOCAL]

    def remote_agents(self) -> list[RegisteredAgent]:
        """Get all remote (A2A) agents."""
        return [a for a in self._agents.values() if a.source == AgentSource.REMOTE]

    def all_cards(self) -> list[AgentCard]:
        """Get all Agent Cards."""
        return [a.card for a in self._agents.values()]

    @property
    def count(self) -> int:
        return len(self._agents)

    # ── System Prompt Awareness ───────────────────────────────────

    def cards_summary(self, budget: int = 4000) -> str:
        """Generate Agent Card summary text for Orchestrator system prompt.

        This is the core mechanism for "Orchestrator understanding downstream agents":
        the summary is injected into the system prompt so the LLM can see all
        available agents and their capabilities on every turn.

        Args:
            budget: Maximum character budget for the summary.

        Returns:
            Formatted summary of all agents, within budget.
        """
        if not self._agents:
            return ""

        lines: list[str] = []
        total_len = 0
        for agent in self._agents.values():
            line = agent.card.to_summary_line()
            if total_len + len(line) + 1 > budget:
                lines.append(f"... and {len(self._agents) - len(lines)} more agents")
                break
            lines.append(f"- {line}")
            total_len += len(line) + 3  # "- " prefix + newline

        return "\n".join(lines)

    def awareness_prompt_block(self, budget: int = 4000) -> str:
        """Generate a complete prompt block for agent awareness.

        Returns a formatted section that can be appended to the system prompt.
        Returns empty string if no agents are registered.
        """
        summary = self.cards_summary(budget)
        if not summary:
            return ""

        return (
            "\n\n## Available Agents\n"
            "The following agents can be invoked via the `skill` tool "
            "or `call_remote_agent` tool:\n"
            f"{summary}\n\n"
            "Route tasks to the agent whose capabilities best match the user's intent. "
            "If a task spans multiple agents, prefer the one with broader coverage.\n"
        )

    # ── Routing ───────────────────────────────────────────────────

    def match(
        self,
        query: str,
        top_k: int = 3,
        min_score: int = 1,
    ) -> list[RegisteredAgent]:
        """Match a user query to the most relevant agents.

        Phase 1 implementation: keyword-based matching against Agent Card text.
        Phase 2 (future): embedding-based semantic matching.

        Args:
            query: User input / intent description.
            top_k: Maximum number of matches to return.
            min_score: Minimum keyword match score to include.

        Returns:
            List of matching agents, sorted by relevance.
        """
        if not query or not self._agents:
            return []

        scored: list[tuple[float, RegisteredAgent]] = []
        query_terms = set(query.lower().split())

        for agent in self._agents.values():
            index_text = agent.card.to_embedding_text().lower()
            index_terms = set(index_text.split())

            # Keyword overlap score
            overlap = query_terms & index_terms
            if not overlap:
                continue

            score = len(overlap)

            # Bonus for name match
            if agent.card.name.lower() in query.lower():
                score += 3

            # Performance weight (Phase 2): boost high-success agents
            if agent.stats.total_calls >= 5:
                score *= 0.5 + 0.5 * agent.stats.success_rate

            if score >= min_score:
                scored.append((score, agent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:top_k]]

    # ── Bulk Operations ───────────────────────────────────────────

    def register_skills(
        self,
        skills: list[Skill],
        base_url: str | None = None,
    ) -> int:
        """Register multiple Skills at once.

        Args:
            skills: List of Skills to register.
            base_url: Optional base URL for A2A exposure.

        Returns:
            Number of agents registered.
        """
        count = 0
        for skill in skills:
            self.register_skill(skill, base_url)
            count += 1
        return count

    def clear(self) -> None:
        """Remove all registered agents."""
        self._agents.clear()
