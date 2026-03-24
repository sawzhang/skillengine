"""
JSONL read/write for session files.

Each session is stored as a JSONL file where the first line is a
:class:`SessionHeader` and subsequent lines are :class:`SessionEntry` objects.
Sessions are organised into directories keyed by a SHA-256 hash of the
working directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skillengine.session.models import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    LabelEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionHeader,
    SessionInfoEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)

# Base directory under the user's home for all session data.
_SESSIONS_BASE = Path.home() / ".skillengine" / "sessions"

# Map from entry ``type`` string to the corresponding dataclass.
_ENTRY_TYPE_MAP: dict[str, type] = {
    "message": SessionMessageEntry,
    "model_change": ModelChangeEntry,
    "thinking_level_change": ThinkingLevelChangeEntry,
    "compaction": CompactionEntry,
    "branch_summary": BranchSummaryEntry,
    "label": LabelEntry,
    "session_info": SessionInfoEntry,
    "custom": CustomEntry,
    "session_header": SessionHeader,
}


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_session_dir(cwd: str) -> Path:
    """
    Return the session storage directory for *cwd*.

    The directory is ``~/.skillengine/sessions/{cwd-hash}/`` where
    *cwd-hash* is the first 16 hex characters of the SHA-256 digest of
    *cwd*.  The directory is created if it does not exist.
    """
    cwd_hash = hashlib.sha256(cwd.encode()).hexdigest()[:16]
    session_dir = _SESSIONS_BASE / cwd_hash
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialize_entry(entry: SessionEntry | SessionHeader) -> str:
    """Serialize a session entry or header to a JSON string."""
    return json.dumps(asdict(entry), separators=(",", ":"))


def _deserialize_entry(line: str) -> SessionEntry | SessionHeader:
    """
    Parse a JSON line into the appropriate typed entry.

    Raises :class:`ValueError` if the line cannot be parsed or has an
    unknown ``type`` field.
    """
    data: dict[str, Any] = json.loads(line)
    entry_type = data.get("type")
    if entry_type is None:
        raise ValueError("Missing 'type' field in session entry")

    cls = _ENTRY_TYPE_MAP.get(entry_type)
    if cls is None:
        raise ValueError(f"Unknown session entry type: {entry_type!r}")

    # Build the dataclass from the JSON dict.  Unknown keys are silently
    # dropped so that forward-compatible fields do not cause errors.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Read / write primitives
# ---------------------------------------------------------------------------


def save_header(path: Path, header: SessionHeader) -> None:
    """
    Write *header* as the first line of the JSONL file at *path*.

    If the file already exists it is **overwritten** (headers are only
    written once, at session creation time).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize_entry(header) + "\n", encoding="utf-8")


def append_entry(path: Path, entry: SessionEntry) -> None:
    """Append *entry* as a new JSONL line to the file at *path*."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_serialize_entry(entry) + "\n")


def load_session(
    path: Path,
) -> tuple[SessionHeader | None, list[SessionEntry]]:
    """
    Read all entries from the JSONL file at *path*.

    Returns ``(header, entries)`` where *header* is the first-line
    :class:`SessionHeader` (or ``None`` if the file is empty / corrupt)
    and *entries* is the list of all subsequent session entries.

    Malformed lines are silently skipped.
    """
    if not path.exists():
        return None, []

    header: SessionHeader | None = None
    entries: list[SessionEntry] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = _deserialize_entry(stripped)
            except (json.JSONDecodeError, ValueError):
                continue

            if isinstance(obj, SessionHeader):
                if header is None:
                    header = obj
                # Ignore duplicate headers (shouldn't happen, but be safe)
            else:
                entries.append(obj)

    return header, entries


def list_sessions(base_dir: Path) -> list[SessionHeader]:
    """
    List all sessions under *base_dir* by reading the header of each
    ``.jsonl`` file.

    Returns a list of :class:`SessionHeader` objects sorted by timestamp
    (most recent first).  Files that cannot be read or lack a valid header
    are silently skipped.
    """
    if not base_dir.is_dir():
        return []

    headers: list[SessionHeader] = []

    for jsonl_file in sorted(base_dir.glob("**/*.jsonl")):
        try:
            with jsonl_file.open("r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            if not first_line:
                continue
            obj = _deserialize_entry(first_line)
            if isinstance(obj, SessionHeader):
                headers.append(obj)
        except (json.JSONDecodeError, ValueError, OSError):
            continue

    # Most recent first
    headers.sort(key=lambda h: h.timestamp, reverse=True)
    return headers
