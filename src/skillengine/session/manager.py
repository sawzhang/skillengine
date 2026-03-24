"""
SessionManager - high-level API for session persistence.

Manages conversation sessions as append-only trees stored in JSONL files.
Each session entry has an ``id`` and ``parent_id`` forming a tree structure.
The *leaf* pointer tracks the current position. Appending creates a child of
the current leaf.  Branching moves the leaf to an earlier entry, allowing new
branches without modifying history.

Mirrors the pi-mono ``SessionManager`` patterns adapted for Python.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from skillengine.session.models import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    LabelEntry,
    ModelChangeEntry,
    SessionContext,
    SessionEntry,
    SessionHeader,
    SessionInfoEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)
from skillengine.session.store import (
    append_entry,
    load_session,
    save_header,
)
from skillengine.session.tree import walk_to_root


class SessionManager:
    """
    Manages a single session's lifecycle: creation, entry appending, tree
    traversal, forking, and navigation.

    The session is persisted as a JSONL file on disk.  In-memory indices
    (``_by_id``) keep lookups fast while the file remains the source of truth.
    """

    def __init__(
        self,
        session_dir: str | Path,
        session_id: str | None = None,
    ) -> None:
        """
        Open or create a session.

        Args:
            session_dir: Directory where session JSONL files are stored.
            session_id: Existing session id to load.  If ``None`` a brand-new
                session is created.
        """
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._header: SessionHeader | None = None
        self._entries: list[SessionEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None

        if session_id is not None:
            self._load_existing(session_id)
        else:
            self._create_new()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _create_new(self) -> None:
        """Create a brand-new session with a fresh header."""
        self._header = SessionHeader(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            cwd=str(self._session_dir),
        )
        self._entries = []
        self._by_id = {}
        self._leaf_id = None
        save_header(self._session_file_path, self._header)

    def _load_existing(self, session_id: str) -> None:
        """Load a session from disk by *session_id*."""
        # Search for a matching JSONL file in the session directory.
        found_path: Path | None = None
        for jsonl_file in self._session_dir.glob("*.jsonl"):
            header, _ = load_session(jsonl_file)
            if header is not None and header.id == session_id:
                found_path = jsonl_file
                break

        if found_path is None:
            raise FileNotFoundError(f"No session with id {session_id!r} in {self._session_dir}")

        header, entries = load_session(found_path)
        assert header is not None  # guaranteed by the search above
        self._header = header
        self._entries = entries
        self._by_id = {e.id: e for e in entries}
        # Default leaf is the last entry appended (file order).
        self._leaf_id = entries[-1].id if entries else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def header(self) -> SessionHeader:
        """Return the session header."""
        assert self._header is not None
        return self._header

    @property
    def entries(self) -> list[SessionEntry]:
        """Return a shallow copy of all session entries (excludes header)."""
        return list(self._entries)

    @property
    def leaf_id(self) -> str | None:
        """The id of the current leaf entry, or ``None`` if the session is empty."""
        return self._leaf_id

    @property
    def _session_file_path(self) -> Path:
        """Path to the JSONL file for this session."""
        assert self._header is not None
        return self._session_dir / f"{self._header.id}.jsonl"

    # ------------------------------------------------------------------
    # Append helpers (private)
    # ------------------------------------------------------------------

    def _append_and_persist(self, entry: SessionEntry) -> None:
        """Add *entry* to internal state and persist to disk."""
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        append_entry(self._session_file_path, entry)

    # ------------------------------------------------------------------
    # Public append methods
    # ------------------------------------------------------------------

    def append_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessageEntry:
        """
        Append a message entry as a child of the current leaf.

        Returns the newly created :class:`SessionMessageEntry`.
        """
        entry = SessionMessageEntry(
            id=str(uuid.uuid4()),
            parent_id=self._leaf_id,
            timestamp=time.time(),
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id,
            name=name,
            metadata=metadata or {},
        )
        self._append_and_persist(entry)
        return entry

    def append_model_change(
        self,
        prev_model: str,
        new_model: str,
        prev_provider: str = "",
        new_provider: str = "",
    ) -> ModelChangeEntry:
        """
        Append a model change entry as a child of the current leaf.

        Returns the newly created :class:`ModelChangeEntry`.
        """
        entry = ModelChangeEntry(
            id=str(uuid.uuid4()),
            parent_id=self._leaf_id,
            timestamp=time.time(),
            previous_model=prev_model,
            new_model=new_model,
            previous_provider=prev_provider,
            new_provider=new_provider,
        )
        self._append_and_persist(entry)
        return entry

    def append_compaction(
        self,
        summary: str,
        tokens_before: int = 0,
        tokens_after: int = 0,
        first_kept_entry_id: str | None = None,
    ) -> CompactionEntry:
        """
        Append a compaction entry as a child of the current leaf.

        Returns the newly created :class:`CompactionEntry`.
        """
        entry = CompactionEntry(
            id=str(uuid.uuid4()),
            parent_id=self._leaf_id,
            timestamp=time.time(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        self._append_and_persist(entry)
        return entry

    def append_thinking_level_change(
        self,
        prev: str,
        new: str,
    ) -> ThinkingLevelChangeEntry:
        """
        Append a thinking-level change entry as a child of the current leaf.

        Returns the newly created :class:`ThinkingLevelChangeEntry`.
        """
        entry = ThinkingLevelChangeEntry(
            id=str(uuid.uuid4()),
            parent_id=self._leaf_id,
            timestamp=time.time(),
            previous_level=prev,
            new_level=new,
        )
        self._append_and_persist(entry)
        return entry

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self) -> SessionContext:
        """
        Walk from the current leaf to the root and build a
        :class:`SessionContext` suitable for LLM consumption.

        Handles compaction: when a :class:`CompactionEntry` is encountered on
        the path, messages before the ``first_kept_entry_id`` are replaced by
        the compaction summary.
        """
        if self._leaf_id is None:
            return SessionContext()

        # Walk leaf -> root, then reverse to get root -> leaf order.
        path = walk_to_root(self._entries, self._leaf_id)
        path.reverse()

        messages: list[SessionMessageEntry] = []
        model_changes: list[ModelChangeEntry] = []
        compactions: list[CompactionEntry] = []
        current_model = ""
        current_thinking_level = "off"

        # Locate the latest compaction on the path (if any).
        compaction: CompactionEntry | None = None
        compaction_idx: int | None = None
        for idx, entry in enumerate(path):
            if entry.type == "compaction":
                assert isinstance(entry, CompactionEntry)
                compaction = entry
                compaction_idx = idx

        if compaction is not None and compaction_idx is not None:
            compactions.append(compaction)

            # Emit kept messages that appear *before* the compaction entry
            # but starting from ``first_kept_entry_id``.
            found_first_kept = False
            for i in range(compaction_idx):
                entry = path[i]
                if (
                    not found_first_kept
                    and compaction.first_kept_entry_id is not None
                    and entry.id == compaction.first_kept_entry_id
                ):
                    found_first_kept = True
                if found_first_kept and isinstance(entry, SessionMessageEntry):
                    messages.append(entry)
                if isinstance(entry, ModelChangeEntry):
                    model_changes.append(entry)
                    current_model = entry.new_model
                if isinstance(entry, ThinkingLevelChangeEntry):
                    current_thinking_level = entry.new_level

            # Emit entries after compaction
            for i in range(compaction_idx + 1, len(path)):
                entry = path[i]
                if isinstance(entry, SessionMessageEntry):
                    messages.append(entry)
                elif isinstance(entry, ModelChangeEntry):
                    model_changes.append(entry)
                    current_model = entry.new_model
                elif isinstance(entry, ThinkingLevelChangeEntry):
                    current_thinking_level = entry.new_level
                elif isinstance(entry, CompactionEntry):
                    compactions.append(entry)
        else:
            # No compaction -- straightforward walk
            for entry in path:
                if isinstance(entry, SessionMessageEntry):
                    messages.append(entry)
                elif isinstance(entry, ModelChangeEntry):
                    model_changes.append(entry)
                    current_model = entry.new_model
                elif isinstance(entry, ThinkingLevelChangeEntry):
                    current_thinking_level = entry.new_level
                elif isinstance(entry, CompactionEntry):
                    compactions.append(entry)

        return SessionContext(
            messages=messages,
            model_changes=model_changes,
            compactions=compactions,
            current_model=current_model,
            current_thinking_level=current_thinking_level,
        )

    # ------------------------------------------------------------------
    # Branching / navigation
    # ------------------------------------------------------------------

    def fork(self, entry_id: str) -> SessionManager:
        """
        Create a new session that branches from *entry_id*.

        The new session contains all entries from root to *entry_id* and
        records the current session as its ``parent_session``.  Returns a
        fresh :class:`SessionManager` for the forked session.
        """
        if entry_id not in self._by_id:
            raise ValueError(f"Entry {entry_id!r} not found in session")

        # Collect the root-to-entry_id path.
        path = walk_to_root(self._entries, entry_id)
        path.reverse()  # root -> entry_id

        # Create the new session.
        new_mgr = SessionManager(session_dir=self._session_dir)
        # Update parentSession in the new header.
        new_mgr._header.parent_session = self.header.id  # type: ignore[union-attr]
        # Re-save header with the parent pointer.
        save_header(new_mgr._session_file_path, new_mgr.header)

        # Re-append entries from the path into the new session.  We
        # re-create them with new ids so the forked session has its own
        # identity, but we preserve the parent chain within the fork.
        id_remap: dict[str, str] = {}
        for entry in path:
            old_id = entry.id
            new_id = str(uuid.uuid4())
            id_remap[old_id] = new_id

            new_parent_id: str | None = None
            if entry.parent_id is not None:
                new_parent_id = id_remap.get(entry.parent_id)

            # Clone entry with remapped ids.  We use a simple approach:
            # copy all fields and replace id / parent_id.
            if isinstance(entry, SessionMessageEntry):
                clone = SessionMessageEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    role=entry.role,
                    content=entry.content,
                    tool_calls=list(entry.tool_calls),
                    tool_call_id=entry.tool_call_id,
                    name=entry.name,
                    metadata=dict(entry.metadata),
                )
            elif isinstance(entry, ModelChangeEntry):
                clone = ModelChangeEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    previous_model=entry.previous_model,
                    new_model=entry.new_model,
                    previous_provider=entry.previous_provider,
                    new_provider=entry.new_provider,
                )
            elif isinstance(entry, ThinkingLevelChangeEntry):
                clone = ThinkingLevelChangeEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    previous_level=entry.previous_level,
                    new_level=entry.new_level,
                )
            elif isinstance(entry, CompactionEntry):
                # Remap first_kept_entry_id if it's in the path.
                remapped_first = None
                if entry.first_kept_entry_id is not None:
                    remapped_first = id_remap.get(
                        entry.first_kept_entry_id, entry.first_kept_entry_id
                    )
                clone = CompactionEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    summary=entry.summary,
                    first_kept_entry_id=remapped_first,
                    tokens_before=entry.tokens_before,
                    tokens_after=entry.tokens_after,
                )
            elif isinstance(entry, BranchSummaryEntry):
                clone = BranchSummaryEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    from_id=id_remap.get(entry.from_id, entry.from_id),
                    summary=entry.summary,
                )
            elif isinstance(entry, LabelEntry):
                clone = LabelEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    target_id=id_remap.get(entry.target_id, entry.target_id),
                    label=entry.label,
                )
            elif isinstance(entry, SessionInfoEntry):
                clone = SessionInfoEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    display_name=entry.display_name,
                )
            elif isinstance(entry, CustomEntry):
                clone = CustomEntry(
                    id=new_id,
                    parent_id=new_parent_id,
                    timestamp=entry.timestamp,
                    custom_type=entry.custom_type,
                    data=dict(entry.data),
                )
            else:
                # Fallback -- should not happen for known types.
                continue

            new_mgr._append_and_persist(clone)

        return new_mgr

    def navigate(self, entry_id: str) -> None:
        """
        Move the leaf pointer to *entry_id*.

        The next ``append_*`` call will create a child of this entry,
        effectively starting a new branch.  Existing entries are never
        modified or deleted.

        Raises :class:`ValueError` if *entry_id* is not in this session.
        """
        if entry_id not in self._by_id:
            raise ValueError(f"Entry {entry_id!r} not found in session")
        self._leaf_id = entry_id
