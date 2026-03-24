"""Event hooks for transparent memory session management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skillengine.logging import get_logger
from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig
from skillengine.memory.tools import MemoryState

if TYPE_CHECKING:
    from skillengine.agent import AgentMessage

logger = get_logger("memory.hooks")


class MemoryHooks:
    """Manages the three lifecycle hooks for memory integration.

    Hooks:
        on_agent_start (priority 10):
            Create an OpenViking session when the agent starts.

        on_context_transform (priority 100):
            Sync new conversation messages to the OV session.

        on_agent_end (priority 90):
            Sync remaining messages and optionally commit for extraction.
    """

    def __init__(
        self,
        client: OpenVikingClient,
        config: MemoryConfig,
        state: MemoryState,
        get_conversation: Any,  # Callable[[], list[AgentMessage]]
    ) -> None:
        self.client = client
        self.config = config
        self.state = state
        self._get_conversation = get_conversation
        self._synced_message_count: int = 0

    async def on_agent_start(self, event: Any) -> None:
        """Create an OV session when the agent starts."""
        if not self.config.auto_session:
            return
        if not self.client.available:
            return

        session_id = await self.client.create_session(
            metadata={"model": getattr(event, "model", "unknown")}
        )
        if session_id:
            self.state.session_id = session_id
            self._synced_message_count = 0
            logger.debug("Created memory session: %s", session_id)
        else:
            logger.warning("Failed to create memory session")

    async def on_context_transform(self, event: Any) -> None:
        """Sync new messages to the OV session."""
        if not self.config.auto_sync:
            return
        if not self.state.session_id:
            return

        conversation: list[AgentMessage] = self._get_conversation()
        new_messages = conversation[self._synced_message_count :]

        for msg in new_messages:
            if msg.role in ("user", "assistant") and msg.content:
                await self.client.add_message(
                    self.state.session_id,
                    msg.role,
                    msg.content,
                )

        self._synced_message_count = len(conversation)

    async def on_agent_end(self, event: Any) -> None:
        """Sync remaining messages and optionally commit."""
        if not self.state.session_id:
            return

        # Sync any remaining messages
        if self.config.auto_sync:
            conversation: list[AgentMessage] = self._get_conversation()
            new_messages = conversation[self._synced_message_count :]

            for msg in new_messages:
                if msg.role in ("user", "assistant") and msg.content:
                    await self.client.add_message(
                        self.state.session_id,
                        msg.role,
                        msg.content,
                    )

            self._synced_message_count = len(conversation)

        # Trigger memory extraction
        if self.config.auto_commit:
            ok = await self.client.commit_session(self.state.session_id)
            if ok:
                logger.debug("Committed memory session: %s", self.state.session_id)
            else:
                logger.warning("Failed to commit memory session: %s", self.state.session_id)
