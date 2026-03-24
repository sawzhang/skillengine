"""OpenViking memory integration for cross-session agent memory."""

from skillengine.memory.client import OpenVikingClient
from skillengine.memory.config import MemoryConfig
from skillengine.memory.extension import setup_memory

__all__ = [
    "MemoryConfig",
    "OpenVikingClient",
    "setup_memory",
]
