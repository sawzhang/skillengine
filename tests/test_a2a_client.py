"""Tests for A2A Client."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from skillengine.a2a.agent_card import AgentCard, AgentCardSkill
from skillengine.a2a.client import A2AClient, create_remote_agent_tool
from skillengine.a2a.models import A2ATaskResponse, TaskStatus
from skillengine.a2a.registry import AgentRegistry


class TestA2AClient:
    def test_init(self):
        client = A2AClient(timeout=30.0)
        assert client.timeout == 30.0

    def test_default_timeout(self):
        client = A2AClient()
        assert client.timeout == 120.0


class TestA2AClientDiscover:
    @pytest.mark.asyncio
    async def test_discover_single_card(self):
        card_data = {
            "name": "remote-agent",
            "description": "A remote agent",
            "version": "1.0.0",
            "skills": [{"name": "analyze", "description": "Analyze data"}],
        }

        client = A2AClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = card_data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            card = await client.discover("http://remote:8080")
            assert card.name == "remote-agent"

    @pytest.mark.asyncio
    async def test_discover_multi_card(self):
        data = {
            "agents": [
                {"name": "agent-a", "description": "Agent A", "skills": []},
                {"name": "agent-b", "description": "Agent B", "skills": []},
            ]
        }

        client = A2AClient()
        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cards = await client.discover_all("http://remote:8080")
            assert len(cards) == 2
            assert cards[0].name == "agent-a"
            assert cards[1].name == "agent-b"


class TestA2AClientSendTask:
    @pytest.mark.asyncio
    async def test_send_task(self):
        resp_data = {
            "task_id": "t1",
            "status": "completed",
            "output": "analysis result",
        }

        client = A2AClient()
        mock_response = MagicMock()
        mock_response.json.return_value = resp_data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            response = await client.send_task(
                endpoint="http://remote:8080",
                skill_name="analyze",
                input_text="test data",
            )
            assert response.status == TaskStatus.COMPLETED
            assert response.output == "analysis result"

            # Verify request was sent correctly
            call_args = mock_ctx.post.call_args
            assert "/tasks" in call_args[0][0]
            sent_data = call_args[1]["json"]
            assert sent_data["skill_name"] == "analyze"


class TestA2AClientDiscoverAndRegister:
    @pytest.mark.asyncio
    async def test_discover_and_register(self):
        data = {
            "agents": [
                {"name": "agent-x", "description": "Agent X", "skills": []},
            ]
        }

        client = A2AClient()
        registry = AgentRegistry()

        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            names = await client.discover_and_register("http://remote:8080", registry)
            assert names == ["agent-x"]
            assert registry.count == 1
            agent = registry.get("agent-x")
            assert agent is not None
            assert agent.endpoint == "http://remote:8080"


class TestCreateRemoteAgentTool:
    def test_no_remote_agents(self):
        registry = AgentRegistry()
        client = A2AClient()
        tool = create_remote_agent_tool(client, registry)

        assert tool["function"]["name"] == "call_remote_agent"
        assert "No remote agents" in tool["function"]["description"]

    def test_with_remote_agents(self):
        registry = AgentRegistry()
        card = AgentCard(name="remote-a", description="Remote Agent A")
        registry.register_remote(card, "http://remote:8080")

        client = A2AClient()
        tool = create_remote_agent_tool(client, registry)

        assert "remote-a" in tool["function"]["description"]
        assert "Remote Agent A" in tool["function"]["description"]
        params = tool["function"]["parameters"]
        assert "agent_name" in params["properties"]
        assert "task" in params["properties"]
