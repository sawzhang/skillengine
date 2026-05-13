"""Lightweight performance baselines for skillengine hot paths.

Run with::

    uv run python -m benchmarks.run

This package is intentionally self-contained — it does not import from
``tests/`` and does not require network access or LLM credentials.
"""

from __future__ import annotations
