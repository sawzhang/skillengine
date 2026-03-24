"""Wiring function to integrate memory into an AgentRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skillengine.events import AGENT_END, AGENT_START, CONTEXT_TRANSFORM
from skillengine.extensions.manager import ExtensionManager
from skillengine.logging import get_logger
from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig
from skillengine.memory.hooks import MemoryHooks
from skillengine.memory.tools import MemoryState, build_memory_tools

if TYPE_CHECKING:
    from skillengine.agent import AgentRunner

logger = get_logger("memory.extension")


async def setup_memory(
    runner: AgentRunner,
    config: MemoryConfig | None = None,
) -> OpenVikingClient | None:
    """Wire OpenViking memory into an AgentRunner.

    1. Creates an ``OpenVikingClient`` and checks health.
    2. If unavailable, logs a warning and returns ``None``.
    3. Initializes ``ExtensionManager`` on the engine if needed.
    4. Registers 4 memory tools via the extension system.
    5. Registers 3 lifecycle hooks on the runner's EventBus.
    6. Returns the client for optional direct use.

    Args:
        runner: The agent runner to integrate memory with.
        config: Memory configuration. Uses defaults if not provided.

    Returns:
        The ``OpenVikingClient`` if available, else ``None``.

    Example::

        runner = AgentRunner.create(system_prompt="You are helpful.")
        client = await setup_memory(runner, MemoryConfig())
        response = await runner.chat("Hello!")
    """
    config = config or MemoryConfig()

    # 1. Create client and health-check
    client = OpenVikingClient(config)
    ok = await client.initialize()
    if not ok:
        logger.warning("OpenViking unavailable at %s — memory disabled", config.base_url)
        return None

    logger.info("OpenViking connected at %s", config.base_url)

    # 2. Ensure ExtensionManager exists
    if runner.engine.extensions is None:
        runner.engine.extensions = ExtensionManager(
            runner.engine,
            event_bus=runner.events,
        )

    ext_manager = runner.engine.extensions

    # 3. Create shared state
    state = MemoryState(client)

    # 4. Register tools
    tools = build_memory_tools(state)
    for tool_info in tools:
        ext_manager._register_tool(
            name=tool_info.name,
            handler=tool_info.handler,
            description=tool_info.description,
            parameters=tool_info.parameters,
            extension_name="memory",
        )

    # 5. Create hooks and register on EventBus
    hooks = MemoryHooks(
        client=client,
        config=config,
        state=state,
        get_conversation=lambda: runner._conversation,
    )

    runner.events.on(AGENT_START, hooks.on_agent_start, priority=10, source="memory")
    runner.events.on(CONTEXT_TRANSFORM, hooks.on_context_transform, priority=100, source="memory")
    runner.events.on(AGENT_END, hooks.on_agent_end, priority=90, source="memory")

    return client
