"""Performance-driven routing for agent selection.

Enhances the basic keyword matching in AgentRegistry with:
1. Performance weighting (success rate, latency)
2. Cost-aware selection
3. Historical preference learning
4. Fallback chains

This is the "dynamic understanding" layer — beyond static Agent Card matching,
the router learns which agents perform best for which types of tasks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from skillengine.a2a.agent_card import AgentCard
from skillengine.a2a.registry import AgentRegistry, AgentStats, RegisteredAgent

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for the performance-aware router."""

    # Weighting factors (must sum to ~1.0 for interpretability)
    keyword_weight: float = 0.4
    performance_weight: float = 0.3
    cost_weight: float = 0.15
    recency_weight: float = 0.15

    # Performance thresholds
    min_calls_for_stats: int = 3  # Min calls before stats affect routing
    latency_penalty_threshold_ms: float = 5000.0  # Start penalizing above this
    failure_rate_penalty: float = 0.5  # Score multiplier for 100% failure rate

    # Fallback
    fallback_enabled: bool = True  # Enable fallback to next-best agent on failure
    max_fallback_depth: int = 2  # Max fallback attempts


@dataclass
class RouteResult:
    """Result of a routing decision."""

    agent: RegisteredAgent
    score: float
    breakdown: dict[str, float]  # Component scores for explainability
    fallbacks: list[RegisteredAgent] = field(default_factory=list)


@dataclass
class _RouteHistory:
    """Tracks routing outcomes for learning."""

    query_pattern: str  # Simplified query pattern
    agent_name: str
    success: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class PerformanceRouter:
    """Performance-aware agent router.

    Scores agents on multiple dimensions:
    1. **Keyword relevance** — how well Agent Card matches the query
    2. **Performance** — historical success rate and latency
    3. **Cost** — agent's declared cost_hint
    4. **Recency** — recent usage patterns

    Usage::

        router = PerformanceRouter(registry)

        # Get best agent for a task
        result = router.route("analyze this tweet for sentiment")
        agent = result.agent
        fallbacks = result.fallbacks  # backup agents if primary fails

        # Record outcome for learning
        router.record_outcome(
            query="analyze this tweet",
            agent_name=agent.card.name,
            success=True,
            latency_ms=1200,
        )
    """

    def __init__(
        self,
        registry: AgentRegistry,
        config: RoutingConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or RoutingConfig()
        self._history: list[_RouteHistory] = []
        self._max_history: int = 1000

    def route(
        self,
        query: str,
        top_k: int = 3,
        exclude: list[str] | None = None,
    ) -> RouteResult | None:
        """Route a query to the best matching agent.

        Args:
            query: User input or task description.
            top_k: Number of fallback candidates.
            exclude: Agent names to exclude (e.g., after failure).

        Returns:
            RouteResult with primary agent and fallbacks, or None if no match.
        """
        if not query or self.registry.count == 0:
            return None

        exclude_set = set(exclude or [])
        scored: list[tuple[float, dict[str, float], RegisteredAgent]] = []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for agent in self.registry.all():
            if agent.card.name in exclude_set:
                continue

            breakdown = self._score_agent(agent, query_lower, query_terms)

            # Require non-zero keyword relevance — without it,
            # neutral performance/cost/recency scores create false matches
            if breakdown.get("keyword", 0) <= 0:
                continue

            total = sum(breakdown.values())
            scored.append((total, breakdown, agent))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)

        primary_score, primary_breakdown, primary = scored[0]
        fallbacks = [a for _, _, a in scored[1:top_k]]

        return RouteResult(
            agent=primary,
            score=primary_score,
            breakdown=primary_breakdown,
            fallbacks=fallbacks,
        )

    def _score_agent(
        self,
        agent: RegisteredAgent,
        query_lower: str,
        query_terms: set[str],
    ) -> dict[str, float]:
        """Score an agent across all dimensions."""
        cfg = self.config
        breakdown: dict[str, float] = {}

        # 1. Keyword relevance
        breakdown["keyword"] = (
            self._keyword_score(agent, query_lower, query_terms) * cfg.keyword_weight
        )

        # 2. Performance (success rate + latency)
        breakdown["performance"] = self._performance_score(agent.stats) * cfg.performance_weight

        # 3. Cost
        breakdown["cost"] = self._cost_score(agent.card) * cfg.cost_weight

        # 4. Recency
        breakdown["recency"] = self._recency_score(agent) * cfg.recency_weight

        return breakdown

    def _keyword_score(
        self,
        agent: RegisteredAgent,
        query_lower: str,
        query_terms: set[str],
    ) -> float:
        """Score based on keyword overlap with Agent Card."""
        index_text = agent.card.to_embedding_text().lower()
        index_terms = set(index_text.split())

        if not query_terms:
            return 0.0

        overlap = query_terms & index_terms
        if not overlap:
            return 0.0

        # Base score: fraction of query terms matched
        score = len(overlap) / len(query_terms)

        # Name match bonus
        if agent.card.name.lower() in query_lower:
            score = min(score + 0.3, 1.0)

        return score

    def _performance_score(self, stats: AgentStats) -> float:
        """Score based on historical performance.

        Returns 0.5 (neutral) when insufficient data.
        Range: 0.0 (always fails, very slow) to 1.0 (always succeeds, fast).
        """
        if stats.total_calls < self.config.min_calls_for_stats:
            return 0.5  # Neutral — not enough data

        # Success rate component (0.0 → 1.0)
        success_score = stats.success_rate

        # Latency component (1.0 for fast, decaying for slow)
        if stats.avg_latency_ms <= 0:
            latency_score = 1.0
        else:
            threshold = self.config.latency_penalty_threshold_ms
            # Sigmoid-like decay: 1.0 at 0ms, ~0.5 at threshold, approaches 0
            latency_score = 1.0 / (1.0 + (stats.avg_latency_ms / threshold) ** 2)

        # Weighted combination (success matters more)
        return 0.7 * success_score + 0.3 * latency_score

    def _cost_score(self, card: AgentCard) -> float:
        """Score based on cost hint. Prefer cheaper agents when quality is similar.

        Returns 0.5 (neutral) if no cost info.
        """
        cost_map = {"low": 1.0, "medium": 0.5, "high": 0.2}
        return cost_map.get(card.cost_hint or "", 0.5)

    def _recency_score(self, agent: RegisteredAgent) -> float:
        """Score based on how recently the agent was successfully used.

        Recently-used agents that succeeded get a boost.
        Range: 0.0 to 1.0
        """
        if agent.stats.last_called is None:
            return 0.5  # Neutral — never used

        age_seconds = time.time() - agent.stats.last_called
        age_hours = age_seconds / 3600.0

        # Exponential decay: full score if used in last hour, halves every 6 hours
        base = math.exp(-age_hours / 6.0)

        # Penalize if last call was a failure
        if agent.stats.last_error:
            base *= 0.3

        return base

    # ── Outcome recording ────────────────────────────────────────

    def record_outcome(
        self,
        query: str,
        agent_name: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record the outcome of a routing decision.

        This feeds the learning loop: future routing decisions
        use these outcomes to improve agent selection.

        Args:
            query: The original query.
            agent_name: The agent that was selected.
            success: Whether the task completed successfully.
            latency_ms: Execution time in milliseconds.
        """
        # Update agent stats
        agent = self.registry.get(agent_name)
        if agent:
            if success:
                agent.stats.record_success(latency_ms)
            else:
                agent.stats.record_failure("task_failure", latency_ms)

        # Store in history (bounded)
        pattern = self._simplify_query(query)
        self._history.append(
            _RouteHistory(
                query_pattern=pattern,
                agent_name=agent_name,
                success=success,
                latency_ms=latency_ms,
            )
        )

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def _simplify_query(self, query: str) -> str:
        """Reduce query to a simplified pattern for history matching.

        Strips specifics, keeps intent keywords.
        """
        # Simple approach: keep only words > 3 chars, lowercase, sorted
        words = sorted(set(w.lower() for w in query.split() if len(w) > 3))
        return " ".join(words[:10])

    # ── Fallback execution ───────────────────────────────────────

    async def route_with_fallback(
        self,
        query: str,
        execute_fn: Any,  # async (RegisteredAgent) -> str
    ) -> tuple[str, RegisteredAgent]:
        """Route and execute with automatic fallback on failure.

        Args:
            query: User query.
            execute_fn: Async function that takes a RegisteredAgent and returns output.

        Returns:
            Tuple of (output, agent_that_succeeded).

        Raises:
            RuntimeError: If all agents (including fallbacks) fail.
        """
        result = self.route(query, top_k=self.config.max_fallback_depth + 1)
        if result is None:
            raise RuntimeError(f"No agents match query: {query!r}")

        candidates = [result.agent] + result.fallbacks
        last_error: Exception | None = None

        for i, agent in enumerate(candidates):
            start = time.time()
            try:
                output = await execute_fn(agent)
                latency_ms = (time.time() - start) * 1000
                self.record_outcome(query, agent.card.name, True, latency_ms)
                if i > 0:
                    logger.info(
                        "Fallback succeeded: %s (attempt %d)",
                        agent.card.name,
                        i + 1,
                    )
                return output, agent
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                self.record_outcome(query, agent.card.name, False, latency_ms)
                last_error = e
                logger.warning(
                    "Agent %s failed (attempt %d/%d): %s",
                    agent.card.name,
                    i + 1,
                    len(candidates),
                    e,
                )

                if not self.config.fallback_enabled:
                    break

        raise RuntimeError(
            f"All {len(candidates)} agents failed for query: {query!r}. Last error: {last_error}"
        )

    # ── Diagnostics ──────────────────────────────────────────────

    def routing_report(self, query: str) -> dict[str, Any]:
        """Generate a detailed routing report for a query (for debugging).

        Shows scores, breakdowns, and ranking for all candidate agents.
        """
        if not query:
            return {"query": query, "candidates": []}

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        candidates = []
        for agent in self.registry.all():
            breakdown = self._score_agent(agent, query_lower, query_terms)
            total = sum(breakdown.values())
            candidates.append(
                {
                    "name": agent.card.name,
                    "source": agent.source.value,
                    "total_score": round(total, 4),
                    "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
                    "stats": {
                        "calls": agent.stats.total_calls,
                        "success_rate": round(agent.stats.success_rate, 2),
                        "avg_latency_ms": round(agent.stats.avg_latency_ms, 1),
                    },
                }
            )

        candidates.sort(key=lambda c: float(c["total_score"]), reverse=True)  # type: ignore[arg-type]

        return {
            "query": query,
            "candidates": candidates,
            "history_size": len(self._history),
        }
