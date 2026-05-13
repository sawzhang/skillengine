"""Built-in span exporters: Console, OpenTelemetry, LangSmith, Logfire."""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any

from .core import Span, SpanExporter, SpanStatus

logger = logging.getLogger(__name__)

__all__ = [
    "ConsoleSpanExporter",
    "OTelSpanExporter",
    "LangSmithSpanExporter",
    "LogfireSpanExporter",
]


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


class ConsoleSpanExporter(SpanExporter):
    """Print each ended span to stderr (or a custom stream) as a JSON line.

    Useful for development. Output format::

        {"name": "agent.turn", "kind": "turn", "duration_ms": 320.1, ...}
    """

    def __init__(self, stream: IO[str] | None = None, *, pretty: bool = False) -> None:
        self.stream = stream or sys.stderr
        self.pretty = pretty

    def export(self, span: Span) -> None:
        payload = _span_to_dict(span)
        try:
            if self.pretty:
                self.stream.write(json.dumps(payload, indent=2, default=str) + "\n")
            else:
                self.stream.write(json.dumps(payload, default=str) + "\n")
            self.stream.flush()
        except Exception:
            logger.exception("ConsoleSpanExporter failed to write span")


# ---------------------------------------------------------------------------
# OpenTelemetry bridge
# ---------------------------------------------------------------------------


class OTelSpanExporter(SpanExporter):
    """Bridge SkillEngine spans into an OpenTelemetry tracer.

    Requires ``opentelemetry-api`` (and an SDK + exporter configured by the
    application). Each ended span becomes a single OTel span with the same
    name, attributes, and events.

    Note: this does *not* preserve full parent-child relationships across OTel
    because SkillEngine spans end out of order with the OTel ``start_as_current``
    contract. For nested timing inside OTel backends use the legacy
    :func:`skillengine.telemetry.install` integration which uses OTel directly.
    """

    def __init__(self, tracer_name: str = "skillengine") -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - exercised w/out otel
            raise RuntimeError(
                "opentelemetry-api is not installed. "
                "Install it with `pip install skillengine[telemetry]`."
            ) from exc
        self._trace = trace
        self._tracer = trace.get_tracer(tracer_name)

    def export(self, span: Span) -> None:
        try:
            otel_span = self._tracer.start_span(span.name, start_time=int(span.start_time * 1e9))
            for k, v in span.attributes.items():
                otel_span.set_attribute(k, _safe_attribute(v))
            for evt in span.events:
                otel_span.add_event(
                    evt["name"],
                    attributes={k: _safe_attribute(v) for k, v in evt["attributes"].items()},
                    timestamp=int(evt.get("timestamp", span.start_time) * 1e9),
                )
            if span.status == SpanStatus.ERROR:
                from opentelemetry.trace import Status, StatusCode

                otel_span.set_status(
                    Status(StatusCode.ERROR, span.attributes.get("error.message", ""))
                )
            end = span.end_time or span.start_time
            otel_span.end(end_time=int(end * 1e9))
        except Exception:
            logger.exception("OTelSpanExporter failed to export span")


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


class LangSmithSpanExporter(SpanExporter):
    """Send spans to LangSmith via ``langsmith.Client.create_run``.

    Requires the ``langsmith`` package and an ``LANGSMITH_API_KEY`` env var.
    Each span is reported as a standalone run. Trace correlation is preserved
    via the ``trace_id`` attribute.
    """

    _KIND_TO_RUN_TYPE = {
        "agent": "chain",
        "turn": "chain",
        "tool": "tool",
        "skill": "tool",
        "compact": "chain",
        "internal": "chain",
    }

    def __init__(self, project_name: str = "skillengine", client: Any | None = None) -> None:
        if client is None:
            try:
                from langsmith import Client  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "langsmith is not installed. `pip install langsmith` first."
                ) from exc
            client = Client()
        self.client = client
        self.project_name = project_name

    def export(self, span: Span) -> None:
        try:
            self.client.create_run(
                name=span.name,
                run_type=self._KIND_TO_RUN_TYPE.get(span.kind.value, "chain"),
                inputs={"attributes": span.attributes},
                outputs={"events": span.events} if span.events else None,
                start_time=span.start_time,
                end_time=span.end_time,
                error=span.attributes.get("error.message")
                if span.status == SpanStatus.ERROR
                else None,
                extra={
                    "trace_id": span.context.trace_id,
                    "parent_span_id": span.context.parent_span_id,
                    "kind": span.kind.value,
                },
                project_name=self.project_name,
            )
        except Exception:
            logger.exception("LangSmithSpanExporter failed to export span")


# ---------------------------------------------------------------------------
# Logfire
# ---------------------------------------------------------------------------


class LogfireSpanExporter(SpanExporter):
    """Send spans to Pydantic Logfire.

    Requires the ``logfire`` package, configured before use
    (``logfire.configure()``). Each span is logged as a single Logfire span
    via ``logfire.span`` with a synthesised duration.
    """

    def __init__(self, logfire_module: Any | None = None) -> None:
        if logfire_module is None:
            try:
                import logfire  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "logfire is not installed. `pip install logfire` first."
                ) from exc
            logfire_module = logfire
        self.logfire = logfire_module

    def export(self, span: Span) -> None:
        try:
            attrs = {k: _safe_attribute(v) for k, v in span.attributes.items()}
            attrs.setdefault("kind", span.kind.value)
            if span.duration_ms is not None:
                attrs["duration_ms"] = span.duration_ms
            level = "error" if span.status == SpanStatus.ERROR else "info"
            self.logfire.log(level, span.name, **attrs)
        except Exception:
            logger.exception("LogfireSpanExporter failed to export span")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span_to_dict(span: Span) -> dict[str, Any]:
    return {
        "name": span.name,
        "kind": span.kind.value,
        "trace_id": span.context.trace_id,
        "span_id": span.context.span_id,
        "parent_span_id": span.context.parent_span_id,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": span.duration_ms,
        "status": span.status.value,
        "attributes": span.attributes,
        "events": span.events,
    }


def _safe_attribute(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)[:512]
