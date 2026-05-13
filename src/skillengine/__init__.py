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
    SummarizingCompactor,
    TokenBudgetCompactor,
    ToolResultTruncator,
    estimate_content_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)
from skillengine.context_files import ContextFile, load_context_files
from skillengine.cost import CostEntry, CostSummary, CostTracker, attach_cost_tracker
from skillengine.definition import AgentDefinition, RuntimeConfig
from skillengine.engine import SkillsEngine
from skillengine.environment import Environment
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
from skillengine.scheduler import CronExpression, CronJob, CronScheduler
from skillengine.typed_output import (
    StructuredOutputError,
    build_directive,
    extract_json_payload,
    extract_json_schema,
    parse_structured,
)

# Optional: MCP integration (always available — no extra deps for v0.3)
try:
    from skillengine.mcp import (
        MCPClient,
        MCPConnectionPool,
        MCPServer,
        MCPServerSpec,
        parse_mcp_uri,
    )
except ImportError:  # pragma: no cover - never expected
    pass

# A2A Handoffs shim (compat with OpenAI Agents SDK / Anthropic A2A draft)
from skillengine.a2a.handoffs import (
    Handoff,
    a2a_handoff,
    agent_handoff,
    callable_handoff,
    handoff,
)

# Eval harness (EVAL-1)
from skillengine.eval import (
    ContainsScorer,
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalReport,
    EvalRunner,
    ExactMatchScorer,
    LLMJudgeScorer,
    RegexScorer,
    Scorer,
    ScorerResult,
    StructuredMatchScorer,
    builtin_suite,
    list_builtin_suites,
)

# Guardrails
from skillengine.guardrails import (
    CostBudgetGuardrail,
    Guardrail,
    GuardrailAction,
    GuardrailManager,
    GuardrailResult,
    GuardrailScope,
    GuardrailViolation,
    PIIGuardrail,
    PromptInjectionGuardrail,
    TokenBudgetGuardrail,
)

# Tracing (TRACE-1)
from skillengine.tracing import (
    ConsoleSpanExporter,
    LangSmithSpanExporter,
    LogfireSpanExporter,
    OTelSpanExporter,
    Span,
    SpanContext,
    SpanExporter,
    SpanKind,
    SpanStatus,
    Tracer,
    install_tracer,
)

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
    "AgentDefinition",
    "RuntimeConfig",
    "Environment",
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
    "SummarizingCompactor",
    "ToolResultTruncator",
    "SlidingWindowCompactor",
    "estimate_tokens",
    "estimate_content_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    # Context Files
    "ContextFile",
    "load_context_files",
    # Cost dashboard (COST-1)
    "CostEntry",
    "CostSummary",
    "CostTracker",
    "attach_cost_tracker",
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
    # Scheduler (cron → prompt)
    "CronExpression",
    "CronJob",
    "CronScheduler",
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
    # Typed / structured output
    "StructuredOutputError",
    "build_directive",
    "extract_json_payload",
    "extract_json_schema",
    "parse_structured",
    # MCP (Model Context Protocol) interop
    "MCPClient",
    "MCPConnectionPool",
    "MCPServer",
    "MCPServerSpec",
    "parse_mcp_uri",
    # A2A handoffs (compat shim with OpenAI Agents SDK / Anthropic A2A draft)
    "Handoff",
    "handoff",
    "callable_handoff",
    "agent_handoff",
    "a2a_handoff",
    # Guardrails
    "Guardrail",
    "GuardrailAction",
    "GuardrailManager",
    "GuardrailResult",
    "GuardrailScope",
    "GuardrailViolation",
    "PIIGuardrail",
    "PromptInjectionGuardrail",
    "TokenBudgetGuardrail",
    "CostBudgetGuardrail",
    # Tracing
    "Span",
    "SpanContext",
    "SpanExporter",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "install_tracer",
    "ConsoleSpanExporter",
    "OTelSpanExporter",
    "LangSmithSpanExporter",
    "LogfireSpanExporter",
    # Eval harness (EVAL-1)
    "EvalCase",
    "EvalDataset",
    "EvalRunner",
    "EvalCaseResult",
    "EvalReport",
    "Scorer",
    "ScorerResult",
    "ExactMatchScorer",
    "ContainsScorer",
    "RegexScorer",
    "StructuredMatchScorer",
    "LLMJudgeScorer",
    "builtin_suite",
    "list_builtin_suites",
]
