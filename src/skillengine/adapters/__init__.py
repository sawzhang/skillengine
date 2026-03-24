"""
LLM provider adapters.

These adapters integrate the skills engine with different LLM providers,
making it easy to use skills in agent workflows.
"""

from skillengine.adapters.base import LLMAdapter
from skillengine.adapters.registry import AdapterFactory, AdapterRegistry

__all__ = ["LLMAdapter", "AdapterRegistry", "AdapterFactory"]

# Optional imports for specific providers
try:
    from skillengine.adapters.openai import OpenAIAdapter  # noqa: F401

    __all__.append("OpenAIAdapter")
except ImportError:
    pass

try:
    from skillengine.adapters.anthropic import AnthropicAdapter  # noqa: F401

    __all__.append("AnthropicAdapter")
except ImportError:
    pass
