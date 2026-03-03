"""
Skill execution runtime.
"""

from skillkit.runtime.base import ExecutionResult, SkillRuntime
from skillkit.runtime.bash import BashRuntime
from skillkit.runtime.code_mode import CodeModeRuntime

__all__ = ["SkillRuntime", "ExecutionResult", "BashRuntime", "CodeModeRuntime"]

# Optional: BoxLite sandbox runtime
try:
    from skillkit.runtime.boxlite import BoxLiteRuntime, SecurityLevel  # noqa: F401

    __all__.extend(["BoxLiteRuntime", "SecurityLevel"])
except ImportError:
    pass
