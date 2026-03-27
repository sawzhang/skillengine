"""
SkillEngine - A standalone skills execution engine for LLM agents.

This library provides a framework for defining, loading, filtering, and executing
skills in LLM-based agent systems. It is designed to be framework-agnostic and
can be integrated with any LLM provider (OpenAI, Anthropic, etc.).

Example:
    from skillengine import SkillsEngine, SkillsConfig

    # Initialize engine
    engine = SkillsEngine(
        config=SkillsConfig(
            skill_dirs=["./skills", "~/.agent/skills"],
            watch=True,
        )
    )

    # Load and filter skills
    skills = engine.load_skills()
    eligible = engine.filter_skills(skills)

    # Generate prompt for LLM
    prompt = engine.format_prompt(eligible)

    # Execute a skill
    result = await engine.execute("github", args={"action": "list-prs"})
"""

from skillengine.adapters.registry import AdapterFactory, AdapterRegistry
from skillengine.agent import (
    AgentAbortedError,
    AgentConfig,
    AgentMessage,
    AgentRunner,
    create_agent,
)
from skillengine.cache import (
    get_cache_config_openai,
    get_cache_control_anthropic,
)
from skillengine.commands import CommandRegistry, CommandResult
from skillengine.config import CacheRetention, SkillEntryConfig, SkillsConfig
from skillengine.context import (
    ContextCompactor,
    ContextManager,
    SlidingWindowCompactor,
    TokenBudgetCompactor,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)
from skillengine.context_files import ContextFile, load_context_files
from skillengine.engine import SkillsEngine
from skillengine.events import (
    AFTER_TOOL_RESULT,
    AGENT_END,
    AGENT_START,
    BEFORE_TOOL_CALL,
    COMPACTION,
    CONTEXT_TRANSFORM,
    INPUT,
    MODEL_CHANGE,
    SESSION_END,
    SESSION_START,
    TOOL_EXECUTION_UPDATE,
    TURN_END,
    TURN_START,
    AfterToolResultEvent,
    AgentEndEvent,
    AgentStartEvent,
    BeforeToolCallEvent,
    CompactionEvent,
    ContextTransformEvent,
    ContextTransformEventResult,
    EventBus,
    InputEvent,
    InputEventResult,
    ModelChangeEvent,
    SessionEndEvent,
    SessionStartEvent,
    StreamEvent,
    ToolCallEventResult,
    ToolExecutionUpdateEvent,
    ToolResultEventResult,
    TurnEndEvent,
    TurnStartEvent,
)
from skillengine.extensions import (
    CommandInfo,
    ExtensionAPI,
    ExtensionInfo,
    ExtensionManager,
    ToolInfo,
)
from skillengine.filters import DefaultSkillFilter, SkillFilter
from skillengine.loaders import MarkdownSkillLoader, SkillLoader
from skillengine.model_registry import (
    DEFAULT_THINKING_BUDGETS,
    CostBreakdown,
    ModelCost,
    ModelDefinition,
    ModelRegistry,
    ThinkingLevel,
    TokenUsage,
    Transport,
    adjust_max_tokens_for_thinking,
    map_thinking_level_to_anthropic_effort,
    map_thinking_level_to_openai_effort,
)
from skillengine.models import (
    ImageContent,
    MessageContent,
    Skill,
    SkillAction,
    SkillActionParam,
    SkillEntry,
    SkillInstallSpec,
    SkillInvocationPolicy,
    SkillMetadata,
    SkillRequirements,
    SkillSnapshot,
    TextContent,
)
from skillengine.prompts import PromptTemplate, PromptTemplateLoader
from skillengine.runtime import BashRuntime, CodeModeRuntime, SkillRuntime

# Optional: BoxLite sandbox runtime
try:
    from skillengine.runtime.boxlite import BoxLiteRuntime, SecurityLevel
except ImportError:
    pass

# Optional: Sandbox module (requires BoxLite)
try:
    from skillengine.sandbox import SandboxedAgentRunner
except ImportError:
    pass

# Optional: memory module
try:
    from skillengine.memory import MemoryConfig, OpenVikingClient, setup_memory
except ImportError:
    pass

# Harness (multi-agent orchestration)
from skillengine.harness import HarnessConfig, HarnessReport, HarnessRunner

# Optimizer (self-improving skill loop)
from skillengine.optimizer import OptimizationReport, OptimizerConfig, SkillOptimizer

try:
    from skillengine._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    # Core models
    "Skill",
    "SkillMetadata",
    "SkillRequirements",
    "SkillSnapshot",
    "SkillEntry",
    "SkillInvocationPolicy",
    "SkillInstallSpec",
    "SkillAction",
    "SkillActionParam",
    # Content types (multi-modal)
    "TextContent",
    "ImageContent",
    "MessageContent",
    # Config
    "SkillsConfig",
    "SkillEntryConfig",
    "CacheRetention",
    # Engine
    "SkillsEngine",
    # Agent
    "AgentRunner",
    "AgentConfig",
    "AgentMessage",
    "AgentAbortedError",
    "create_agent",
    # Events
    "EventBus",
    "AGENT_START",
    "AGENT_END",
    "TURN_START",
    "TURN_END",
    "BEFORE_TOOL_CALL",
    "AFTER_TOOL_RESULT",
    "CONTEXT_TRANSFORM",
    "INPUT",
    "TOOL_EXECUTION_UPDATE",
    "SESSION_START",
    "SESSION_END",
    "MODEL_CHANGE",
    "COMPACTION",
    "ToolExecutionUpdateEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "BeforeToolCallEvent",
    "ToolCallEventResult",
    "AfterToolResultEvent",
    "ToolResultEventResult",
    "ContextTransformEvent",
    "ContextTransformEventResult",
    "InputEvent",
    "InputEventResult",
    "StreamEvent",
    "SessionStartEvent",
    "SessionEndEvent",
    "ModelChangeEvent",
    "CompactionEvent",
    # Model Registry
    "ModelDefinition",
    "ModelCost",
    "ModelRegistry",
    "TokenUsage",
    "CostBreakdown",
    # Thinking & Transport
    "ThinkingLevel",
    "Transport",
    "DEFAULT_THINKING_BUDGETS",
    "adjust_max_tokens_for_thinking",
    "map_thinking_level_to_anthropic_effort",
    "map_thinking_level_to_openai_effort",
    # Context Management
    "ContextManager",
    "ContextCompactor",
    "TokenBudgetCompactor",
    "SlidingWindowCompactor",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    # Context Files
    "ContextFile",
    "load_context_files",
    # Cache
    "get_cache_control_anthropic",
    "get_cache_config_openai",
    # Loaders
    "SkillLoader",
    "MarkdownSkillLoader",
    # Filters
    "SkillFilter",
    "DefaultSkillFilter",
    # Runtime
    "SkillRuntime",
    "BashRuntime",
    "CodeModeRuntime",
    "BoxLiteRuntime",
    "SecurityLevel",
    "SandboxedAgentRunner",
    # Adapters
    "AdapterRegistry",
    "AdapterFactory",
    # Extensions
    "ExtensionAPI",
    "ExtensionManager",
    "ExtensionInfo",
    "CommandInfo",
    "ToolInfo",
    # Commands
    "CommandRegistry",
    "CommandResult",
    # Prompts
    "PromptTemplate",
    "PromptTemplateLoader",
    # Memory (optional)
    "MemoryConfig",
    "OpenVikingClient",
    "setup_memory",
    # Harness (multi-agent orchestration)
    "HarnessRunner",
    "HarnessConfig",
    "HarnessReport",
    # Optimizer (self-improving skill loop)
    "SkillOptimizer",
    "OptimizerConfig",
    "OptimizationReport",
]
