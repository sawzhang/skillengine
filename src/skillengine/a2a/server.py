"""A2A Server — expose SkillEngine agents as A2A endpoints.

Endpoints:
- GET  /.well-known/agent.json  → Aggregated Agent Cards
- POST /tasks                   → Create and execute a task
- GET  /tasks/{task_id}         → Query task status
- POST /tasks/{task_id}/cancel  → Cancel a task

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skillengine.a2a.models import A2ATaskRequest, A2ATaskResponse, TaskStatus
from skillengine.a2a.registry import AgentRegistry

if TYPE_CHECKING:
    from skillengine.agent import AgentConfig
    from skillengine.engine import SkillsEngine

logger = logging.getLogger(__name__)


@dataclass
class _TaskRecord:
    """In-memory task state."""

    task_id: str
    skill_name: str
    input_text: str
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class A2AServer:
    """Expose SkillEngine Skills as A2A-compatible HTTP endpoints.

    Usage::

        from skillengine import SkillsEngine, SkillsConfig
        from skillengine.a2a import AgentRegistry
        from skillengine.a2a.server import A2AServer

        engine = SkillsEngine(config=SkillsConfig(skill_dirs=["./skills"]))
        registry = AgentRegistry()
        registry.register_skills(engine.load_skills())

        server = A2AServer(engine=engine, registry=registry)
        app = server.create_app()

        # Run with: uvicorn app:app
    """

    def __init__(
        self,
        engine: SkillsEngine,
        registry: AgentRegistry,
        agent_config: AgentConfig | None = None,
        server_name: str = "skillengine",
        server_version: str = "0.1.0",
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.agent_config = agent_config
        self.server_name = server_name
        self.server_version = server_version
        self._tasks: dict[str, _TaskRecord] = {}

    def create_app(self) -> Any:
        """Create a FastAPI application with A2A endpoints.

        Returns:
            A FastAPI app instance.

        Raises:
            ImportError: If fastapi is not installed.
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for A2A server. "
                "Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title=f"{self.server_name} A2A Server",
            version=self.server_version,
        )

        @app.get("/.well-known/agent.json")
        async def agent_card() -> JSONResponse:
            """Return aggregated Agent Cards for all exposed Skills."""
            exposed = []
            for agent in self.registry.all():
                # Only expose agents with a2a.expose=True, or all local if no filter
                skill = agent.skill
                if skill is None:
                    continue
                a2a_config = getattr(skill, "_a2a_config", {})
                if a2a_config.get("expose", False):
                    exposed.append(agent.card.to_dict())

            # If no agents explicitly expose, expose all local agents
            if not exposed:
                exposed = [a.card.to_dict() for a in self.registry.local_agents()]

            return JSONResponse(
                content={
                    "server": self.server_name,
                    "version": self.server_version,
                    "agents": exposed,
                }
            )

        @app.post("/tasks")
        async def create_task(request: dict) -> JSONResponse:
            """Create and execute an A2A task."""
            try:
                task_req = A2ATaskRequest.from_dict(request)
            except (KeyError, TypeError) as e:
                raise HTTPException(400, f"Invalid request: {e}")

            agent = self.registry.get(task_req.skill_name)
            if agent is None or agent.skill is None:
                available = [a.card.name for a in self.registry.local_agents()]
                raise HTTPException(
                    404,
                    f"Agent '{task_req.skill_name}' not found. "
                    f"Available: {available}",
                )

            # Create task record
            record = _TaskRecord(
                task_id=task_req.task_id,
                skill_name=task_req.skill_name,
                input_text=task_req.input_text,
                status=TaskStatus.IN_PROGRESS,
            )
            self._tasks[record.task_id] = record

            # Execute skill
            start_time = time.time()
            try:
                output = await self._execute_skill(agent.skill, task_req.input_text)
                latency_ms = (time.time() - start_time) * 1000

                record.status = TaskStatus.COMPLETED
                record.output = output
                record.completed_at = time.time()

                agent.stats.record_success(latency_ms)

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                record.status = TaskStatus.FAILED
                record.error = str(e)
                record.completed_at = time.time()

                agent.stats.record_failure(str(e), latency_ms)
                logger.exception("Task %s failed", record.task_id)

            response = A2ATaskResponse(
                task_id=record.task_id,
                status=record.status,
                output=record.output,
                error=record.error,
            )
            return JSONResponse(content=response.to_dict())

        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str) -> JSONResponse:
            """Query task status."""
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            response = A2ATaskResponse(
                task_id=record.task_id,
                status=record.status,
                output=record.output,
                error=record.error,
            )
            return JSONResponse(content=response.to_dict())

        @app.post("/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str) -> JSONResponse:
            """Cancel a running task."""
            record = self._tasks.get(task_id)
            if record is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            if record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                raise HTTPException(
                    400,
                    f"Task already {record.status.value}",
                )

            record._cancel_event.set()
            record.status = TaskStatus.CANCELLED
            record.completed_at = time.time()

            return JSONResponse(
                content=A2ATaskResponse(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                ).to_dict()
            )

        @app.get("/health")
        async def health() -> dict:
            return {
                "status": "ok",
                "agents": self.registry.count,
                "tasks_total": len(self._tasks),
            }

        return app

    async def _execute_skill(self, skill: Any, input_text: str) -> str:
        """Execute a skill and return its output.

        Uses AgentRunner in fork mode for isolation.
        """
        from skillengine.agent import AgentConfig, AgentRunner

        config = self.agent_config or AgentConfig.from_env()
        runner = AgentRunner(engine=self.engine, config=config)

        # Execute skill content as system prompt, input as user message
        response = await runner.chat(input_text)
        return response.text_content if hasattr(response, "text_content") else str(response.content)

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the A2A server (blocking).

        Args:
            host: Bind address.
            port: Listen port.
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn is required. Install with: pip install uvicorn")

        app = self.create_app()
        uvicorn.run(app, host=host, port=port)
