"""A2A Coordinator Agent — central entry point for multi-agent systems.

The CoordinatorAgent acts as both an A2A server and client:
- **As a server**: accepts tasks from external callers at POST /tasks
- **As a client**: connects to downstream remote agents via A2A discovery
- **As a router**: selects the best agent using PerformanceRouter

Topology::

    External Caller
         │  POST /tasks
         ▼
    CoordinatorAgent  (port 8000)
    ├── AgentServer A (port 8001) — e.g. "weather" skills
    ├── AgentServer B (port 8002) — e.g. "code-review" skills
    └── AgentServer C (port 8003) — e.g. "math" skills

Routing strategy:
    1. If ``skill_name`` exactly matches a registered agent → route directly
    2. Otherwise → PerformanceRouter scores all agents by keyword + perf + cost
    3. Send task via A2AClient to the winning agent's endpoint
    4. Record outcome for future learning
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skillengine.a2a.client import A2AClient
from skillengine.a2a.models import A2ATaskRequest, A2ATaskResponse, TaskStatus
from skillengine.a2a.registry import AgentRegistry
from skillengine.a2a.router import PerformanceRouter, RoutingConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for the CoordinatorAgent."""

    name: str = "coordinator"
    description: str = (
        "Central coordinator that routes tasks to specialized downstream agents "
        "based on their Agent Card capabilities."
    )
    version: str = "0.1.0"
    author: str | None = None
    tags: list[str] = field(default_factory=lambda: ["coordinator", "orchestrator", "router"])

    # Downstream agent endpoints to connect on startup
    remote_endpoints: list[str] = field(default_factory=list)

    # Routing configuration
    routing_config: RoutingConfig | None = None

    # HTTP client timeout for downstream calls
    client_timeout: float = 120.0

    # Timeout for initial agent discovery on startup
    connect_timeout: float = 10.0


@dataclass
class _CoordTaskRecord:
    """In-memory task state for the coordinator."""

    task_id: str
    input_text: str
    routed_to: str | None = None  # agent name that handled it
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class CoordinatorAgent:
    """Central coordinator agent that routes tasks to specialized downstream agents.

    Usage::

        config = CoordinatorConfig(
            name="my-coordinator",
            remote_endpoints=[
                "http://agent-a:8001",
                "http://agent-b:8002",
            ],
        )
        coordinator = CoordinatorAgent(config)

        # Connect to downstream agents (discovers their agent cards)
        await coordinator.connect_all()

        # Start the coordinator server
        coordinator.run(host="0.0.0.0", port=8000)

    Or connect dynamically at runtime via POST /agents/connect.
    """

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config or CoordinatorConfig()
        self.registry = AgentRegistry()
        self.client = A2AClient(timeout=self.config.client_timeout)
        self.router = PerformanceRouter(
            self.registry,
            config=self.config.routing_config,
        )
        self._tasks: dict[str, _CoordTaskRecord] = {}

    # ── Agent Discovery ───────────────────────────────────────────────────

    async def connect(self, endpoint: str) -> list[str]:
        """Connect to a remote agent server and register all its agents.

        Fetches ``/.well-known/agent.json`` and registers each agent card.

        Args:
            endpoint: Base URL of the remote A2A server.

        Returns:
            List of registered agent names.

        Raises:
            httpx.HTTPError: If the endpoint is unreachable.
        """
        client = A2AClient(timeout=self.config.connect_timeout)
        names = await client.discover_and_register(endpoint, self.registry)
        logger.info(
            "Connected to %s → registered agents: %s",
            endpoint,
            names,
        )
        return names

    async def connect_all(self) -> dict[str, list[str]]:
        """Connect to all endpoints in ``config.remote_endpoints``.

        Returns:
            Dict mapping endpoint URL → list of registered agent names.
        """
        results: dict[str, list[str]] = {}
        for endpoint in self.config.remote_endpoints:
            try:
                names = await self.connect(endpoint)
                results[endpoint] = names
            except Exception as e:
                logger.warning("Failed to connect to %s: %s", endpoint, e)
                results[endpoint] = []
        return results

    # ── Task Routing ──────────────────────────────────────────────────────

    async def route_task(self, task_req: A2ATaskRequest) -> A2ATaskResponse:
        """Route a task to the best downstream agent and return its response.

        Routing logic:
        1. If ``skill_name`` exactly matches a registered agent → use it directly
        2. Otherwise → PerformanceRouter selects best agent from ``input_text``
        3. Task is forwarded via A2AClient to the agent's endpoint
        4. Outcome is recorded for future routing improvement

        Args:
            task_req: The incoming A2A task request.

        Returns:
            The downstream agent's response.

        Raises:
            ValueError: If no agents are registered or no match found.
        """
        if self.registry.count == 0:
            raise ValueError(
                "No downstream agents registered. "
                "Call connect() or POST /agents/connect first."
            )

        # --- Step 1: Select agent ---
        agent = self.registry.get(task_req.skill_name)
        if agent is not None:
            logger.debug(
                "Direct route: skill_name=%r → agent=%r",
                task_req.skill_name,
                agent.card.name,
            )
        else:
            # Use performance router on input_text
            route_result = self.router.route(task_req.input_text)
            if route_result is None:
                # Fallback: if nothing matches by content, try skill_name as a fuzzy hint
                route_result = self.router.route(task_req.skill_name)
            if route_result is None:
                available = [a.card.name for a in self.registry.all()]
                raise ValueError(
                    f"No agent matches query {task_req.input_text!r}. "
                    f"Available agents: {available}"
                )
            agent = route_result.agent
            logger.info(
                "Routed %r → %s (score=%.3f, breakdown=%s)",
                task_req.input_text[:60],
                agent.card.name,
                route_result.score,
                {k: round(v, 3) for k, v in route_result.breakdown.items()},
            )

        # --- Step 2: Forward to downstream agent ---
        if agent.endpoint is None:
            raise ValueError(
                f"Agent {agent.card.name!r} has no endpoint (it may be local-only). "
                "Only remote agents can be called via coordinator routing."
            )

        start_time = time.time()
        try:
            response = await self.client.send_task(
                endpoint=agent.endpoint,
                skill_name=agent.card.name,
                input_text=task_req.input_text,
                metadata={
                    **task_req.metadata,
                    "coordinator_task_id": task_req.task_id,
                    "routed_by": self.config.name,
                },
            )
            latency_ms = (time.time() - start_time) * 1000
            self.router.record_outcome(task_req.input_text, agent.card.name, True, latency_ms)
            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.router.record_outcome(task_req.input_text, agent.card.name, False, latency_ms)
            raise RuntimeError(
                f"Downstream agent {agent.card.name!r} failed: {e}"
            ) from e

    # ── Agent Card ────────────────────────────────────────────────────────

    def build_card(self, base_url: str | None = None) -> dict[str, Any]:
        """Build this coordinator's own Agent Card.

        The card describes the coordinator itself, and lists all downstream
        agents as its skills — giving callers a single discovery point.

        Args:
            base_url: The coordinator's own URL (set by create_app).

        Returns:
            Agent Card dict in A2A format.
        """
        downstream_skills = []
        for agent in self.registry.all():
            downstream_skills.append(
                {
                    "name": agent.card.name,
                    "description": agent.card.description,
                    "tags": agent.card.tags,
                }
            )

        card: dict[str, Any] = {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "tags": self.config.tags,
            "capabilities": {
                "streaming": False,
                "multi_turn": False,
                "push_notifications": False,
            },
            "skills": downstream_skills,
            "agent_count": self.registry.count,
        }
        if base_url:
            card["url"] = base_url
        if self.config.author:
            card["author"] = self.config.author

        return card

    # ── FastAPI App ───────────────────────────────────────────────────────

    def create_app(self, base_url: str | None = None) -> Any:
        """Create a FastAPI application exposing the coordinator as A2A server.

        Endpoints:
            GET  /.well-known/agent.json  → coordinator's own Agent Card
            POST /tasks                   → receive and route tasks
            GET  /tasks/{task_id}         → query task status
            POST /tasks/{task_id}/cancel  → cancel a task (best-effort)
            GET  /health                  → health + stats
            GET  /agents                  → list all connected downstream agents
            POST /agents/connect          → dynamically connect to a new endpoint

        Args:
            base_url: The public URL of this coordinator (included in agent card).

        Returns:
            A FastAPI app instance.

        Raises:
            ImportError: If fastapi is not installed.
        """
        try:
            from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
            from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "FastAPI is required. Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title=f"{self.config.name} — A2A Coordinator",
            description=self.config.description,
            version=self.config.version,
        )

        # ── Discovery endpoint ────────────────────────────────────────────

        @app.get("/.well-known/agent.json")
        async def agent_card() -> JSONResponse:
            """Return this coordinator's Agent Card."""
            return JSONResponse(content=self.build_card(base_url))

        # ── Task endpoints ────────────────────────────────────────────────

        @app.post("/tasks")
        async def create_task(request: dict[str, Any]) -> JSONResponse:
            """Receive a task, route it to the best downstream agent, return result."""
            try:
                task_req = A2ATaskRequest.from_dict(request)
            except (KeyError, TypeError) as e:
                raise HTTPException(400, f"Invalid request: {e}")

            record = _CoordTaskRecord(
                task_id=task_req.task_id,
                input_text=task_req.input_text,
                status=TaskStatus.IN_PROGRESS,
            )
            self._tasks[record.task_id] = record

            try:
                response = await self.route_task(task_req)
                record.status = response.status
                record.output = response.output
                record.error = response.error
                record.routed_to = response.metadata.get("routed_to", task_req.skill_name)
                record.completed_at = time.time()

                # Surface routing info in metadata
                result_metadata = dict(response.metadata)
                result_metadata["coordinator"] = self.config.name
                result_metadata["coordinator_task_id"] = record.task_id

                return JSONResponse(
                    content=A2ATaskResponse(
                        task_id=record.task_id,
                        status=record.status,
                        output=record.output,
                        error=record.error,
                        metadata=result_metadata,
                    ).to_dict()
                )

            except ValueError as e:
                record.status = TaskStatus.FAILED
                record.error = str(e)
                record.completed_at = time.time()
                raise HTTPException(404, str(e))

            except RuntimeError as e:
                record.status = TaskStatus.FAILED
                record.error = str(e)
                record.completed_at = time.time()
                raise HTTPException(502, str(e))

        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str) -> JSONResponse:
            """Query coordinator task status."""
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            return JSONResponse(
                content=A2ATaskResponse(
                    task_id=record.task_id,
                    status=record.status,
                    output=record.output,
                    error=record.error,
                    metadata={"routed_to": record.routed_to} if record.routed_to else {},
                ).to_dict()
            )

        @app.post("/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str) -> JSONResponse:
            """Cancel a coordinator task (best-effort — downstream may not support it)."""
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            if record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                raise HTTPException(400, f"Task already {record.status.value}")

            record.status = TaskStatus.CANCELLED
            record.completed_at = time.time()

            return JSONResponse(
                content=A2ATaskResponse(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                ).to_dict()
            )

        # ── Health ────────────────────────────────────────────────────────

        @app.get("/health")
        async def health() -> dict[str, Any]:
            """Health check with agent and task stats."""
            agents_info = []
            for agent in self.registry.all():
                agents_info.append({
                    "name": agent.card.name,
                    "source": agent.source.value,
                    "endpoint": agent.endpoint,
                    "calls": agent.stats.total_calls,
                    "success_rate": round(agent.stats.success_rate, 2),
                    "avg_latency_ms": round(agent.stats.avg_latency_ms, 1),
                })

            task_counts = {s.value: 0 for s in TaskStatus}
            for t in self._tasks.values():
                task_counts[t.status.value] += 1

            return {
                "status": "ok",
                "coordinator": self.config.name,
                "agents_connected": self.registry.count,
                "agents": agents_info,
                "tasks": task_counts,
            }

        # ── Agent management ──────────────────────────────────────────────

        @app.get("/agents")
        async def list_agents() -> JSONResponse:
            """List all connected downstream agents with their cards."""
            agents = []
            for agent in self.registry.all():
                agents.append({
                    "name": agent.card.name,
                    "description": agent.card.description,
                    "source": agent.source.value,
                    "endpoint": agent.endpoint,
                    "tags": agent.card.tags,
                    "version": agent.card.version,
                    "stats": {
                        "calls": agent.stats.total_calls,
                        "success_rate": round(agent.stats.success_rate, 2),
                        "avg_latency_ms": round(agent.stats.avg_latency_ms, 1),
                        "last_error": agent.stats.last_error,
                    },
                })
            return JSONResponse(content={"agents": agents, "count": len(agents)})

        @app.post("/agents/connect")
        async def connect_agent(request: dict[str, Any]) -> JSONResponse:
            """Dynamically connect to a new downstream agent endpoint.

            Request body: ``{"endpoint": "http://agent-host:port"}``
            """
            endpoint = request.get("endpoint")
            if not endpoint:
                raise HTTPException(400, "Missing 'endpoint' field")

            try:
                names = await self.connect(str(endpoint))
                return JSONResponse(
                    content={"connected": names, "endpoint": endpoint}
                )
            except Exception as e:
                raise HTTPException(502, f"Failed to connect to {endpoint}: {e}")

        @app.delete("/agents/{agent_name}")
        async def disconnect_agent(agent_name: str) -> JSONResponse:
            """Remove a downstream agent from the registry."""
            removed = self.registry.unregister(agent_name)
            if not removed:
                raise HTTPException(404, f"Agent '{agent_name}' not found")
            return JSONResponse(content={"removed": agent_name})

        return app

    # ── Blocking runner ───────────────────────────────────────────────────

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        base_url: str | None = None,
        connect_on_start: bool = True,
    ) -> None:
        """Start the coordinator server (blocking).

        If ``connect_on_start=True`` (default), connects to all
        ``config.remote_endpoints`` before serving.

        Args:
            host: Bind address.
            port: Listen port.
            base_url: The coordinator's public URL for its agent card.
            connect_on_start: Whether to discover remote agents before starting.
        """
        try:
            import uvicorn  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("uvicorn is required. Install with: pip install uvicorn")

        if connect_on_start and self.config.remote_endpoints:
            asyncio.run(self.connect_all())

        resolved_base_url = base_url or f"http://{host}:{port}"
        app = self.create_app(base_url=resolved_base_url)
        uvicorn.run(app, host=host, port=port)
