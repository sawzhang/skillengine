"""Tests for A2A Server."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from skillengine.a2a.models import A2ATaskRequest, A2ATaskResponse, TaskStatus
from skillengine.a2a.registry import AgentRegistry
from skillengine.models import Skill, SkillMetadata


def _make_skill(name: str, description: str, expose: bool = False) -> Skill:
    skill = Skill(
        name=name,
        description=description,
        content=f"# {name}\nDo the thing.",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        metadata=SkillMetadata(),
    )
    if expose:
        skill._a2a_config = {"expose": True}
    return skill


class TestA2AModels:
    def test_task_request_from_dict(self):
        data = {
            "skill_name": "read-tweet",
            "input_text": "https://x.com/test/status/123",
            "task_id": "abc123",
        }
        req = A2ATaskRequest.from_dict(data)
        assert req.skill_name == "read-tweet"
        assert req.input_text == "https://x.com/test/status/123"
        assert req.task_id == "abc123"

    def test_task_request_to_dict(self):
        req = A2ATaskRequest(
            skill_name="test",
            input_text="hello",
            task_id="t1",
        )
        d = req.to_dict()
        assert d["skill_name"] == "test"
        assert d["task_id"] == "t1"

    def test_task_request_auto_id(self):
        req = A2ATaskRequest(skill_name="test", input_text="hello")
        assert len(req.task_id) == 12

    def test_task_response_roundtrip(self):
        resp = A2ATaskResponse(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            output="result here",
        )
        d = resp.to_dict()
        restored = A2ATaskResponse.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.status == TaskStatus.COMPLETED
        assert restored.output == "result here"

    def test_task_response_with_error(self):
        resp = A2ATaskResponse(
            task_id="t1",
            status=TaskStatus.FAILED,
            error="timeout",
        )
        d = resp.to_dict()
        assert d["error"] == "timeout"

    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestA2AServerCreation:
    """Test server creation without FastAPI dependency."""

    def test_server_init(self):
        from skillengine.a2a.server import A2AServer

        engine = MagicMock()
        registry = AgentRegistry()
        server = A2AServer(engine=engine, registry=registry)

        assert server.engine is engine
        assert server.registry is registry
        assert server.server_name == "skillengine"

    def test_server_custom_config(self):
        from skillengine.a2a.server import A2AServer

        server = A2AServer(
            engine=MagicMock(),
            registry=AgentRegistry(),
            server_name="my-server",
            server_version="2.0.0",
        )
        assert server.server_name == "my-server"
        assert server.server_version == "2.0.0"


class TestA2AServerApp:
    """Test server app creation (requires FastAPI)."""

    @pytest.fixture
    def server(self):
        from skillengine.a2a.server import A2AServer

        engine = MagicMock()
        registry = AgentRegistry()
        skill = _make_skill("test-skill", "A test skill", expose=True)
        registry.register_skill(skill)

        return A2AServer(engine=engine, registry=registry)

    def test_create_app(self, server):
        try:
            app = server.create_app()
            assert app is not None
            # Check routes exist
            routes = [r.path for r in app.routes]
            assert "/.well-known/agent.json" in routes
            assert "/tasks" in routes
            assert "/health" in routes
        except ImportError:
            pytest.skip("FastAPI not installed")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, server):
        try:
            from httpx import ASGITransport, AsyncClient
        except ImportError:
            pytest.skip("httpx ASGITransport not available")

        try:
            app = server.create_app()
        except ImportError:
            pytest.skip("FastAPI not installed")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["agents"] == 1

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, server):
        try:
            from httpx import ASGITransport, AsyncClient
        except ImportError:
            pytest.skip("httpx ASGITransport not available")

        try:
            app = server.create_app()
        except ImportError:
            pytest.skip("FastAPI not installed")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/.well-known/agent.json")
            assert resp.status_code == 200
            data = resp.json()
            assert "agents" in data
            assert len(data["agents"]) >= 1
            assert data["agents"][0]["name"] == "test-skill"
