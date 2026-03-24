"""Session data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SessionEntryType = Literal[
    "message",
    "model_change",
    "thinking_level_change",
    "compaction",
    "branch_summary",
    "label",
    "session_info",
    "custom",
]


@dataclass
class SessionHeader:
    """Metadata for a stored session."""

    type: str = "session_header"
    version: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    cwd: str = ""
    parent_session: str | None = None  # For forked sessions


@dataclass
class SessionMessageEntry:
    """A message entry in the session."""

    type: str = "message"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    role: str = ""  # user, assistant, tool, system
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelChangeEntry:
    """Records a model change in the session."""

    type: str = "model_change"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    previous_model: str = ""
    new_model: str = ""
    previous_provider: str = ""
    new_provider: str = ""


@dataclass
class ThinkingLevelChangeEntry:
    """Records a thinking level change."""

    type: str = "thinking_level_change"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    previous_level: str = "off"
    new_level: str = "off"


@dataclass
class CompactionEntry:
    """Records a context compaction event."""

    type: str = "compaction"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    summary: str = ""
    first_kept_entry_id: str | None = None
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass
class BranchSummaryEntry:
    """Summary of a branched conversation."""

    type: str = "branch_summary"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    from_id: str = ""  # Entry ID where the branch diverged
    summary: str = ""


@dataclass
class LabelEntry:
    """A user-defined bookmark on an entry."""

    type: str = "label"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    target_id: str = ""
    label: str = ""


@dataclass
class SessionInfoEntry:
    """Session metadata like display name."""

    type: str = "session_info"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    display_name: str = ""


@dataclass
class CustomEntry:
    """Extension-specific non-LLM data."""

    type: str = "custom"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    custom_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# Union of all session entry types
SessionEntry = (
    SessionMessageEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | LabelEntry
    | SessionInfoEntry
    | CustomEntry
)


@dataclass
class SessionContext:
    """Built context from session entries for LLM consumption."""

    messages: list[SessionMessageEntry] = field(default_factory=list)
    model_changes: list[ModelChangeEntry] = field(default_factory=list)
    compactions: list[CompactionEntry] = field(default_factory=list)
    current_model: str = ""
    current_thinking_level: str = "off"
