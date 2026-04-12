"""Execution environment for agents.

Wraps the runtime, extensions, adapters, and tool dispatcher into a
single unit. Corresponds to the Managed Agents 'Environment' concept —
the 'Hands' that an agent can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skillengine.adapters.registry import AdapterRegistry
from skillengine.context import ContextManager
from skillengine.engine import SkillsEngine
from skillengine.events import EventBus
from skillengine.model_registry import ModelRegistry
from skillengine.tools.dispatcher import ToolDispatcher


@dataclass
class Environment:
    """Execution environment wrapping runtime, extensions, and adapters.

    Represents what the agent CAN DO — the tools, models, and
    infrastructure available to it.
    """

    engine: SkillsEngine
    adapter_registry: AdapterRegistry = field(default_factory=AdapterRegistry)
    model_registry: ModelRegistry | None = None
    context_manager: ContextManager | None = None
    event_bus: EventBus = field(default_factory=EventBus)
    dispatcher: ToolDispatcher | None = None
