"""Sandbox module — provides SandboxedAgentRunner for BoxLite isolation."""

from __future__ import annotations

try:
    from skillengine.sandbox.runner import SandboxedAgentRunner  # noqa: F401

    __all__ = ["SandboxedAgentRunner"]
except ImportError:
    # boxlite not installed — sandbox features unavailable
    __all__: list[str] = []  # type: ignore[no-redef]
