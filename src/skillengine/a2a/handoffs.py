"""A2A-1: OpenAI Agents SDK "Handoffs" compatibility shim.

Provides a small adapter layer that wraps remote A2A agents, local
``AgentRunner`` instances, or arbitrary async callables as LLM tools using
the *Handoffs* ergonomic from the OpenAI Agents SDK.

The pattern: the LLM calls a tool named ``transfer_to_<agent>``; the tool
forwards a payload to the target agent and returns its final output. An
optional ``input_filter`` can transform the message history before transfer,
matching the OpenAI Agents SDK ``handoff(input_filter=...)`` extension point.

Anthropic's draft A2A protocol uses the same conceptual model — a tool whose
side effect is to delegate to another agent — so the same shim works for
both ecosystems.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from skillengine.a2a.client import A2AClient
from skillengine.tools.registry import ToolDefinition

__all__ = [
    "Handoff",
    "HandoffTarget",
    "InputFilter",
    "agent_handoff",
    "a2a_handoff",
    "callable_handoff",
    "handoff",
    "to_tool_definition",
]


# A target may return a string synchronously or asynchronously.
HandoffCallable = Callable[[str, dict[str, Any]], Any]
InputFilter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


@runtime_checkable
class HandoffTarget(Protocol):
    """Anything that can receive a handoff and return a string output."""

    async def __call__(  # pragma: no cover - structural only
        self, input_text: str, context: dict[str, Any]
    ) -> str: ...


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    s = _SLUG_RE.sub("_", value.lower()).strip("_")
    return s or "agent"


@dataclass
class Handoff:
    """A configured handoff from the current agent to a downstream target.

    Attributes:
        tool_name: Name of the synthetic tool exposed to the LLM. Defaults to
            ``transfer_to_<slug>``.
        description: Tool description shown to the LLM.
        target: Async callable accepting ``(input_text, context)``.
        input_filter: Optional callable applied to message history before
            transfer (mirrors the OpenAI Agents SDK extension point).
        on_handoff: Optional hook fired before the target is invoked.
        input_schema: JSON schema for the tool's ``arguments`` payload.
    """

    tool_name: str
    description: str
    target: HandoffCallable
    input_filter: InputFilter | None = None
    on_handoff: Callable[[dict[str, Any]], Any] | None = None
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The task or message to forward to the target agent.",
                },
            },
            "required": ["input"],
        }
    )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def handoff(
    target: HandoffCallable,
    *,
    name: str,
    description: str | None = None,
    tool_name: str | None = None,
    input_filter: InputFilter | None = None,
    on_handoff: Callable[[dict[str, Any]], Any] | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Handoff:
    """Wrap an arbitrary async callable as a Handoff.

    Args:
        target: Async callable accepting ``(input_text, context)``.
        name: Logical agent name (used to derive ``tool_name``).
        description: Tool description. Defaults to ``"Transfer to {name}"``.
        tool_name: Override for the tool name. Defaults to
            ``transfer_to_<slug(name)>``.
        input_filter: Optional message-history filter.
        on_handoff: Optional pre-call hook.
        input_schema: Optional JSON schema for the arguments.
    """
    slug = _slug(name)
    tool = tool_name or f"transfer_to_{slug}"
    desc = description or f"Transfer the conversation to {name}."
    kwargs: dict[str, Any] = {}
    if input_schema is not None:
        kwargs["input_schema"] = input_schema
    return Handoff(
        tool_name=tool,
        description=desc,
        target=target,
        input_filter=input_filter,
        on_handoff=on_handoff,
        **kwargs,
    )


def callable_handoff(
    target: Callable[[str], str] | Callable[[str], Awaitable[str]],
    *,
    name: str,
    description: str | None = None,
    **kwargs: Any,
) -> Handoff:
    """Wrap a simple ``f(input_text) -> str`` (sync or async) as a Handoff."""

    async def _adapter(input_text: str, context: dict[str, Any]) -> str:
        result = target(input_text)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    return handoff(_adapter, name=name, description=description, **kwargs)


def agent_handoff(
    agent: Any,
    *,
    name: str,
    description: str | None = None,
    reset: bool = True,
    **kwargs: Any,
) -> Handoff:
    """Wrap a local :class:`AgentRunner` as a Handoff.

    The handoff invokes ``agent.chat(input_text, reset=reset)`` and returns
    the final assistant text.
    """

    async def _adapter(input_text: str, context: dict[str, Any]) -> str:
        message = await agent.chat(input_text, reset=reset)
        content = getattr(message, "content", message)
        return content if isinstance(content, str) else str(content)

    return handoff(_adapter, name=name, description=description, **kwargs)


def a2a_handoff(
    client: A2AClient,
    *,
    endpoint: str,
    skill_name: str,
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Handoff:
    """Wrap a remote A2A agent's skill as a Handoff."""

    async def _adapter(input_text: str, context: dict[str, Any]) -> str:
        response = await client.send_task(
            endpoint=endpoint,
            skill_name=skill_name,
            input_text=input_text,
            metadata=context.get("metadata") if context else None,
        )
        if response.output:
            return response.output
        if response.error:
            return f"[remote error] {response.error}"
        return ""

    return handoff(
        _adapter,
        name=name or skill_name,
        description=description or f"Transfer to remote A2A agent '{skill_name}'.",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tool-definition adapter
# ---------------------------------------------------------------------------


def to_tool_definition(
    h: Handoff,
    *,
    context: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Convert a Handoff into a SkillEngine :class:`ToolDefinition`.

    The returned definition's handler accepts the SkillEngine dispatcher
    signature ``(name, args, ctx, on_output)`` and returns the target's
    output as a string.
    """
    base_context = dict(context or {})

    async def _handler(
        _name: str,
        args: dict[str, Any] | None,
        ctx: Any = None,
        on_output: Any = None,
    ) -> str:
        args = args or {}
        input_text = str(args.get("input") or args.get("input_text") or "")
        merged_ctx = dict(base_context)
        if isinstance(ctx, dict):
            merged_ctx.update(ctx)
        # Apply input_filter if a message history is supplied in context.
        if h.input_filter is not None and isinstance(merged_ctx.get("messages"), list):
            merged_ctx["messages"] = h.input_filter(list(merged_ctx["messages"]))
        if h.on_handoff is not None:
            result = h.on_handoff(args)
            if inspect.isawaitable(result):
                await result
        output = await h.target(input_text, merged_ctx)
        return output if isinstance(output, str) else str(output)

    return ToolDefinition(
        name=h.tool_name,
        description=h.description,
        parameters=h.input_schema,
        handler=_handler,
    )
