"""A2A Client — discover and invoke remote A2A agents.

Integrates with AgentRegistry for automatic discovery,
and provides a tool definition for Orchestrator LLM routing.

Requires: httpx (already a dependency)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from skillkit.a2a.agent_card import AgentCard
from skillkit.a2a.models import A2ATaskRequest, A2ATaskResponse, TaskStatus
from skillkit.a2a.registry import AgentRegistry

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for discovering and invoking remote A2A agents.

    Usage::

        client = A2AClient()

        # Discover a remote agent
        card = await client.discover("http://remote-agent:8080")

        # Register it in the registry
        registry.register_remote(card, endpoint="http://remote-agent:8080")

        # Send a task
        response = await client.send_task(
            endpoint="http://remote-agent:8080",
            skill_name="analyze",
            input_text="Analyze this data...",
        )
    """

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout

    async def discover(self, endpoint: str) -> AgentCard:
        """Discover a remote agent by fetching its Agent Card.

        Args:
            endpoint: Base URL of the remote A2A agent.

        Returns:
            The remote agent's AgentCard.

        Raises:
            httpx.HTTPError: If the request fails.
            KeyError: If the response is malformed.
        """
        url = f"{endpoint.rstrip('/')}/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # Handle both single-card and multi-card responses
        if "agents" in data and isinstance(data["agents"], list):
            if not data["agents"]:
                raise ValueError(f"No agents found at {endpoint}")
            # Return first agent card
            return AgentCard.from_dict(data["agents"][0])
        else:
            return AgentCard.from_dict(data)

    async def discover_all(self, endpoint: str) -> list[AgentCard]:
        """Discover all agents at a remote endpoint.

        Args:
            endpoint: Base URL of the remote A2A server.

        Returns:
            List of AgentCards.
        """
        url = f"{endpoint.rstrip('/')}/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if "agents" in data and isinstance(data["agents"], list):
            return [AgentCard.from_dict(a) for a in data["agents"]]
        else:
            return [AgentCard.from_dict(data)]

    async def send_task(
        self,
        endpoint: str,
        skill_name: str,
        input_text: str,
        metadata: dict | None = None,
    ) -> A2ATaskResponse:
        """Send a task to a remote A2A agent.

        Args:
            endpoint: Base URL of the remote agent.
            skill_name: Name of the skill/agent to invoke.
            input_text: Task input.
            metadata: Optional metadata.

        Returns:
            Task response with output or error.
        """
        url = f"{endpoint.rstrip('/')}/tasks"
        request = A2ATaskRequest(
            skill_name=skill_name,
            input_text=input_text,
            metadata=metadata or {},
        )

        start_time = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=request.to_dict())
            resp.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        response = A2ATaskResponse.from_dict(resp.json())

        logger.debug(
            "A2A task %s → %s/%s: %s (%.0fms)",
            response.task_id,
            endpoint,
            skill_name,
            response.status.value,
            latency_ms,
        )
        return response

    async def get_task_status(
        self,
        endpoint: str,
        task_id: str,
    ) -> A2ATaskResponse:
        """Query the status of a previously submitted task.

        Args:
            endpoint: Base URL of the remote agent.
            task_id: The task ID to query.

        Returns:
            Current task status and output.
        """
        url = f"{endpoint.rstrip('/')}/tasks/{task_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        return A2ATaskResponse.from_dict(resp.json())

    async def cancel_task(
        self,
        endpoint: str,
        task_id: str,
    ) -> A2ATaskResponse:
        """Cancel a running task on a remote agent.

        Args:
            endpoint: Base URL of the remote agent.
            task_id: The task ID to cancel.

        Returns:
            Updated task status.
        """
        url = f"{endpoint.rstrip('/')}/tasks/{task_id}/cancel"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url)
            resp.raise_for_status()

        return A2ATaskResponse.from_dict(resp.json())

    async def discover_and_register(
        self,
        endpoint: str,
        registry: AgentRegistry,
    ) -> list[str]:
        """Discover remote agents and register them in the registry.

        Convenience method that combines discover + register.

        Args:
            endpoint: Base URL of the remote A2A server.
            registry: The AgentRegistry to register into.

        Returns:
            List of registered agent names.
        """
        cards = await self.discover_all(endpoint)
        names = []
        for card in cards:
            registry.register_remote(card, endpoint=endpoint)
            names.append(card.name)
            logger.info("Discovered and registered remote agent: %s", card.name)
        return names


def create_remote_agent_tool(
    client: A2AClient,
    registry: AgentRegistry,
) -> dict[str, Any]:
    """Create an LLM tool definition for calling remote agents.

    This tool is injected into the Orchestrator's tool list,
    allowing the LLM to route tasks to remote A2A agents.

    Args:
        client: The A2A client for making requests.
        registry: The agent registry with remote agents.

    Returns:
        OpenAI function-calling format tool definition.
    """
    remote = registry.remote_agents()
    if not remote:
        agent_desc = "No remote agents currently available."
    else:
        lines = []
        for agent in remote:
            lines.append(f"- {agent.card.name}: {agent.card.description}")
        agent_desc = "Available remote agents:\n" + "\n".join(lines)

    return {
        "type": "function",
        "function": {
            "name": "call_remote_agent",
            "description": (
                "Call a remote A2A agent to perform a task. "
                "Use this when the task requires capabilities "
                "not available locally.\n\n" + agent_desc
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the remote agent to call.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description / input for the agent.",
                    },
                },
                "required": ["agent_name", "task"],
            },
        },
    }
