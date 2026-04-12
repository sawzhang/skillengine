"""Agent definition and runtime configuration.

Separates the immutable identity of an agent (what it IS) from the
mutable runtime configuration (how it runs NOW). Aligns with the
Managed Agents pattern of reusable agent definitions across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillengine.model_registry import ThinkingLevel, Transport


@dataclass(frozen=True)
class AgentDefinition:
    """Immutable definition of an agent — what it IS.

    Reusable across sessions. Can be serialized and stored.
    Corresponds to the Managed Agents 'Agent' concept.

    Fields here represent the agent's identity and capabilities,
    not how it connects to providers or how many turns it runs.
    """

    # Identity
    name: str = "default"
    description: str = ""

    # LLM identity
    model: str = "MiniMax-M2.1"
    system_prompt: str = ""

    # Capabilities
    skill_dirs: tuple[Path, ...] = ()
    enable_tools: bool = True
    enable_reasoning: bool = True

    # Context
    load_context_files: bool = True
    skill_description_budget: int = 16000


@dataclass
class RuntimeConfig:
    """Mutable runtime configuration — how the agent runs NOW.

    Can change between sessions or mid-session. Contains provider
    connection details, execution limits, and transport settings.
    """

    # LLM connection
    base_url: str | None = None
    api_key: str | None = None
    oauth_token: str | None = None

    # Runtime behavior
    max_turns: int = 20
    auto_execute: bool = True
    temperature: float = 1.0
    max_tokens: int = 4096

    # Thinking
    thinking_level: ThinkingLevel | None = None

    # Transport
    transport: Transport = "sse"

    # Cache
    cache_retention: str = "short"
    session_id: str | None = None

    # Skill watching
    watch_skills: bool = False
