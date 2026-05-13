"""End-to-end tracing for SkillEngine.

This package provides a framework-agnostic span model plus a small set of
exporters (console, OpenTelemetry, LangSmith, Logfire). The span schema is:

============= ================================================
Span name     Emitted when…
============= ================================================
``agent.run`` an :class:`AgentRunner.chat` call begins
``agent.turn`` each LLM round-trip starts (one per turn)
``tool.call``  a tool call is dispatched (one per call)
``skill.load`` a skill body is loaded on demand
``compact.run`` a context compactor reduces history
============= ================================================

Standard attributes (set when available):

* ``model`` — current LLM model
* ``tokens.input`` / ``tokens.output`` / ``tokens.total``
* ``cost_usd``
* ``cache.hit`` — boolean
* ``abort`` — boolean (true if user aborted)
* ``error`` / ``error.message``

Use :func:`install_tracer` to wire a :class:`Tracer` into an agent's
:class:`~skillengine.events.EventBus`. Use :meth:`Tracer.start_span` directly
from custom code (e.g. inside a custom runtime).
"""

from .core import (
    Span,
    SpanContext,
    SpanExporter,
    SpanKind,
    SpanStatus,
    Tracer,
    install_tracer,
)
from .exporters import (
    ConsoleSpanExporter,
    LangSmithSpanExporter,
    LogfireSpanExporter,
    OTelSpanExporter,
)

__all__ = [
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
]
