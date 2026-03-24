"""
Skill execution runtime.
"""

from skillengine.runtime.base import ExecutionResult, SkillRuntime
from skillengine.runtime.bash import BashRuntime
from skillengine.runtime.code_mode import CodeModeRuntime

__all__ = ["SkillRuntime", "ExecutionResult", "BashRuntime", "CodeModeRuntime"]

# Optional: BoxLite sandbox runtime
try:
    from skillengine.runtime.boxlite import BoxLiteRuntime, SecurityLevel  # noqa: F401

    __all__.extend(["BoxLiteRuntime", "SecurityLevel"])
except ImportError:
    pass
