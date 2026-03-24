"""Tests for session module."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

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
    _deserialize_entry,
    _serialize_entry,
    append_entry,
    get_session_dir,
    list_sessions,
    load_session,
    save_header,
)
from skillengine.session.tree import (
    SessionTreeNode,
    build_tree,
    find_entry,
    get_branches,
    walk_to_root,
)
from skillengine.session.manager import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    *,
    id: str = "",
    parent_id: str | None = None,
    role: str = "user",
    content: str = "",
    timestamp: float | None = None,
) -> SessionMessageEntry:
    """Create a SessionMessageEntry with explicit id for testing."""
    return SessionMessageEntry(
        id=id or str(time.time()),
        parent_id=parent_id,
        timestamp=timestamp or time.time(),
        role=role,
        content=content,
    )


# ===================================================================
# TestSessionModels
# ===================================================================


class TestSessionModels:
    """Tests for session data models."""

    def test_session_header_defaults(self) -> None:
        """SessionHeader should have sensible defaults."""
        header = SessionHeader()
        assert header.type == "session_header"
        assert header.version == 1
        assert header.id  # non-empty uuid
        assert header.cwd == ""
        assert header.parent_session is None
        assert header.timestamp > 0

    def test_session_header_custom_values(self) -> None:
        """SessionHeader should accept custom values."""
        header = SessionHeader(
            id="custom-id",
            cwd="/home/user/project",
            parent_session="parent-id",
        )
        assert header.id == "custom-id"
        assert header.cwd == "/home/user/project"
        assert header.parent_session == "parent-id"

    def test_session_message_entry_creation(self) -> None:
        """SessionMessageEntry should be created with all fields."""
        entry = SessionMessageEntry(
            role="user",
            content="Hello, world!",
            tool_calls=[{"id": "tc1", "type": "function"}],
            tool_call_id="tc1",
            name="test-tool",
            metadata={"key": "value"},
        )
        assert entry.type == "message"
        assert entry.role == "user"
        assert entry.content == "Hello, world!"
        assert entry.id  # auto-generated uuid
        assert entry.parent_id is None
        assert entry.timestamp > 0
        assert len(entry.tool_calls) == 1
        assert entry.tool_call_id == "tc1"
        assert entry.name == "test-tool"
        assert entry.metadata == {"key": "value"}

    def test_session_message_entry_defaults(self) -> None:
        """SessionMessageEntry defaults should be empty."""
        entry = SessionMessageEntry()
        assert entry.role == ""
        assert entry.content == ""
        assert entry.tool_calls == []
        assert entry.tool_call_id is None
        assert entry.name is None
        assert entry.metadata == {}

    def test_all_entry_types_have_common_fields(self) -> None:
        """Every entry type should have id, parent_id, timestamp, and type."""
        entry_classes = [
            SessionMessageEntry,
            ModelChangeEntry,
            ThinkingLevelChangeEntry,
            CompactionEntry,
            BranchSummaryEntry,
            LabelEntry,
            SessionInfoEntry,
            CustomEntry,
        ]
        for cls in entry_classes:
            entry = cls()
            assert hasattr(entry, "id"), f"{cls.__name__} missing id"
            assert hasattr(entry, "parent_id"), f"{cls.__name__} missing parent_id"
            assert hasattr(entry, "timestamp"), f"{cls.__name__} missing timestamp"
            assert hasattr(entry, "type"), f"{cls.__name__} missing type"
            assert entry.id, f"{cls.__name__} id should be non-empty"
            assert entry.timestamp > 0, f"{cls.__name__} timestamp should be > 0"
            assert entry.parent_id is None, f"{cls.__name__} parent_id should default to None"

    def test_model_change_entry(self) -> None:
        """ModelChangeEntry should record model transitions."""
        entry = ModelChangeEntry(
            previous_model="gpt-4",
            new_model="claude-3",
            previous_provider="openai",
            new_provider="anthropic",
        )
        assert entry.type == "model_change"
        assert entry.previous_model == "gpt-4"
        assert entry.new_model == "claude-3"
        assert entry.previous_provider == "openai"
        assert entry.new_provider == "anthropic"

    def test_thinking_level_change_entry(self) -> None:
        """ThinkingLevelChangeEntry should record thinking level transitions."""
        entry = ThinkingLevelChangeEntry(previous_level="off", new_level="high")
        assert entry.type == "thinking_level_change"
        assert entry.previous_level == "off"
        assert entry.new_level == "high"

    def test_compaction_entry(self) -> None:
        """CompactionEntry should record compaction events."""
        entry = CompactionEntry(
            summary="Summarized conversation",
            first_kept_entry_id="entry-5",
            tokens_before=10000,
            tokens_after=2000,
        )
        assert entry.type == "compaction"
        assert entry.summary == "Summarized conversation"
        assert entry.first_kept_entry_id == "entry-5"
        assert entry.tokens_before == 10000
        assert entry.tokens_after == 2000

    def test_branch_summary_entry(self) -> None:
        """BranchSummaryEntry should record branch summaries."""
        entry = BranchSummaryEntry(from_id="entry-3", summary="Branch summary")
        assert entry.type == "branch_summary"
        assert entry.from_id == "entry-3"
        assert entry.summary == "Branch summary"

    def test_label_entry(self) -> None:
        """LabelEntry should record labels on entries."""
        entry = LabelEntry(target_id="entry-7", label="checkpoint")
        assert entry.type == "label"
        assert entry.target_id == "entry-7"
        assert entry.label == "checkpoint"

    def test_session_info_entry(self) -> None:
        """SessionInfoEntry should record session metadata."""
        entry = SessionInfoEntry(display_name="My Session")
        assert entry.type == "session_info"
        assert entry.display_name == "My Session"

    def test_custom_entry(self) -> None:
        """CustomEntry should hold arbitrary data."""
        entry = CustomEntry(
            custom_type="analytics",
            data={"event": "click", "count": 5},
        )
        assert entry.type == "custom"
        assert entry.custom_type == "analytics"
        assert entry.data == {"event": "click", "count": 5}

    def test_session_context_defaults(self) -> None:
        """SessionContext should have empty defaults."""
        ctx = SessionContext()
        assert ctx.messages == []
        assert ctx.model_changes == []
        assert ctx.compactions == []
        assert ctx.current_model == ""
        assert ctx.current_thinking_level == "off"

    def test_unique_ids_across_instances(self) -> None:
        """Each entry instance should have a unique id by default."""
        entries = [SessionMessageEntry() for _ in range(10)]
        ids = [e.id for e in entries]
        assert len(set(ids)) == 10


# ===================================================================
# TestSessionStore
# ===================================================================


class TestSessionStore:
    """Tests for session JSONL store."""

    def test_save_header_and_load(self, tmp_path: Path) -> None:
        """save_header should write a header that load_session can read back."""
        path = tmp_path / "session.jsonl"
        header = SessionHeader(id="test-session", cwd="/tmp/project")
        save_header(path, header)

        loaded_header, entries = load_session(path)

        assert loaded_header is not None
        assert loaded_header.id == "test-session"
        assert loaded_header.cwd == "/tmp/project"
        assert loaded_header.type == "session_header"
        assert entries == []

    def test_save_header_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_header should create parent directories if they do not exist."""
        path = tmp_path / "nested" / "dir" / "session.jsonl"
        header = SessionHeader(id="nested-session")
        save_header(path, header)

        assert path.exists()
        loaded_header, _ = load_session(path)
        assert loaded_header is not None
        assert loaded_header.id == "nested-session"

    def test_append_entry(self, tmp_path: Path) -> None:
        """append_entry should add entries that load_session recovers."""
        path = tmp_path / "session.jsonl"
        header = SessionHeader(id="test-session")
        save_header(path, header)

        msg1 = SessionMessageEntry(id="msg-1", role="user", content="Hello")
        msg2 = SessionMessageEntry(
            id="msg-2", parent_id="msg-1", role="assistant", content="Hi there"
        )
        append_entry(path, msg1)
        append_entry(path, msg2)

        loaded_header, entries = load_session(path)
        assert loaded_header is not None
        assert len(entries) == 2
        assert entries[0].id == "msg-1"
        assert entries[1].id == "msg-2"
        assert isinstance(entries[0], SessionMessageEntry)
        assert isinstance(entries[1], SessionMessageEntry)
        assert entries[0].content == "Hello"
        assert entries[1].content == "Hi there"
        assert entries[1].parent_id == "msg-1"

    def test_append_entry_different_types(self, tmp_path: Path) -> None:
        """append_entry should handle all entry types correctly."""
        path = tmp_path / "session.jsonl"
        save_header(path, SessionHeader(id="multi-type"))

        msg = SessionMessageEntry(id="e1", role="user", content="test")
        mc = ModelChangeEntry(
            id="e2", parent_id="e1", previous_model="a", new_model="b"
        )
        comp = CompactionEntry(
            id="e3", parent_id="e2", summary="compacted", tokens_before=100
        )
        tlc = ThinkingLevelChangeEntry(
            id="e4", parent_id="e3", previous_level="off", new_level="high"
        )
        custom = CustomEntry(
            id="e5", parent_id="e4", custom_type="ext", data={"x": 1}
        )

        for entry in [msg, mc, comp, tlc, custom]:
            append_entry(path, entry)

        _, entries = load_session(path)
        assert len(entries) == 5
        assert isinstance(entries[0], SessionMessageEntry)
        assert isinstance(entries[1], ModelChangeEntry)
        assert isinstance(entries[2], CompactionEntry)
        assert isinstance(entries[3], ThinkingLevelChangeEntry)
        assert isinstance(entries[4], CustomEntry)
        assert entries[4].data == {"x": 1}

    def test_load_session_nonexistent_file(self, tmp_path: Path) -> None:
        """load_session should return (None, []) for missing files."""
        path = tmp_path / "nonexistent.jsonl"
        header, entries = load_session(path)
        assert header is None
        assert entries == []

    def test_load_session_with_malformed_lines(self, tmp_path: Path) -> None:
        """load_session should silently skip malformed JSON lines."""
        path = tmp_path / "session.jsonl"
        header = SessionHeader(id="test-session")
        save_header(path, header)

        # Append a valid entry, then a malformed line, then another valid entry
        msg = SessionMessageEntry(id="msg-1", role="user", content="ok")
        append_entry(path, msg)

        with path.open("a", encoding="utf-8") as fh:
            fh.write("this is not valid json\n")
            fh.write('{"type":"unknown_xyz","id":"bad"}\n')
            fh.write("\n")  # empty line

        msg2 = SessionMessageEntry(id="msg-2", role="assistant", content="ok2")
        append_entry(path, msg2)

        loaded_header, entries = load_session(path)
        assert loaded_header is not None
        assert loaded_header.id == "test-session"
        assert len(entries) == 2
        assert entries[0].id == "msg-1"
        assert entries[1].id == "msg-2"

    def test_load_session_empty_file(self, tmp_path: Path) -> None:
        """load_session should handle an empty file."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        header, entries = load_session(path)
        assert header is None
        assert entries == []

    def test_list_sessions(self, tmp_path: Path) -> None:
        """list_sessions should return headers sorted by timestamp (newest first)."""
        # Create two sessions with different timestamps
        s1_path = tmp_path / "s1.jsonl"
        s2_path = tmp_path / "s2.jsonl"

        h1 = SessionHeader(id="session-1", timestamp=1000.0)
        h2 = SessionHeader(id="session-2", timestamp=2000.0)

        save_header(s1_path, h1)
        save_header(s2_path, h2)

        headers = list_sessions(tmp_path)
        assert len(headers) == 2
        assert headers[0].id == "session-2"  # newer first
        assert headers[1].id == "session-1"

    def test_list_sessions_empty_dir(self, tmp_path: Path) -> None:
        """list_sessions should return empty list for a directory with no sessions."""
        headers = list_sessions(tmp_path)
        assert headers == []

    def test_list_sessions_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_sessions should return empty list for nonexistent directory."""
        headers = list_sessions(tmp_path / "does-not-exist")
        assert headers == []

    def test_list_sessions_skips_corrupt_files(self, tmp_path: Path) -> None:
        """list_sessions should skip files that cannot be parsed."""
        good_path = tmp_path / "good.jsonl"
        bad_path = tmp_path / "bad.jsonl"

        save_header(good_path, SessionHeader(id="good-session"))
        bad_path.write_text("not json at all\n", encoding="utf-8")

        headers = list_sessions(tmp_path)
        assert len(headers) == 1
        assert headers[0].id == "good-session"

    def test_get_session_dir_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_session_dir should create the directory if it does not exist."""
        # Monkeypatch the base directory so we don't pollute the real home
        monkeypatch.setattr(
            "skillengine.session.store._SESSIONS_BASE",
            tmp_path / "sessions",
        )
        session_dir = get_session_dir("/some/project")
        assert session_dir.is_dir()
        assert session_dir.parent == tmp_path / "sessions"

    def test_get_session_dir_deterministic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_session_dir should return the same directory for the same cwd."""
        monkeypatch.setattr(
            "skillengine.session.store._SESSIONS_BASE",
            tmp_path / "sessions",
        )
        dir1 = get_session_dir("/my/project")
        dir2 = get_session_dir("/my/project")
        assert dir1 == dir2

    def test_get_session_dir_different_for_different_cwds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_session_dir should return different directories for different cwds."""
        monkeypatch.setattr(
            "skillengine.session.store._SESSIONS_BASE",
            tmp_path / "sessions",
        )
        dir1 = get_session_dir("/project/a")
        dir2 = get_session_dir("/project/b")
        assert dir1 != dir2

    def test_serialize_deserialize_roundtrip(self) -> None:
        """_serialize_entry and _deserialize_entry should round-trip correctly."""
        entry = SessionMessageEntry(
            id="rt-1",
            parent_id="rt-0",
            role="user",
            content="Round trip test",
            metadata={"nested": {"key": "value"}},
        )
        serialized = _serialize_entry(entry)
        deserialized = _deserialize_entry(serialized)

        assert isinstance(deserialized, SessionMessageEntry)
        assert deserialized.id == "rt-1"
        assert deserialized.parent_id == "rt-0"
        assert deserialized.role == "user"
        assert deserialized.content == "Round trip test"
        assert deserialized.metadata == {"nested": {"key": "value"}}

    def test_serialize_deserialize_header_roundtrip(self) -> None:
        """Header serialization and deserialization should round-trip."""
        header = SessionHeader(
            id="hdr-1", cwd="/test", parent_session="parent-1"
        )
        serialized = _serialize_entry(header)
        deserialized = _deserialize_entry(serialized)

        assert isinstance(deserialized, SessionHeader)
        assert deserialized.id == "hdr-1"
        assert deserialized.cwd == "/test"
        assert deserialized.parent_session == "parent-1"

    def test_deserialize_unknown_type_raises(self) -> None:
        """_deserialize_entry should raise ValueError for unknown types."""
        bad_json = json.dumps({"type": "nonexistent_type", "id": "x"})
        with pytest.raises(ValueError, match="Unknown session entry type"):
            _deserialize_entry(bad_json)

    def test_deserialize_missing_type_raises(self) -> None:
        """_deserialize_entry should raise ValueError when type field is missing."""
        bad_json = json.dumps({"id": "x", "content": "no type"})
        with pytest.raises(ValueError, match="Missing 'type' field"):
            _deserialize_entry(bad_json)

    def test_deserialize_ignores_unknown_fields(self) -> None:
        """_deserialize_entry should silently drop unknown fields."""
        data = {
            "type": "message",
            "id": "uf-1",
            "role": "user",
            "content": "test",
            "future_field": "should be ignored",
            "timestamp": 1234.0,
        }
        entry = _deserialize_entry(json.dumps(data))
        assert isinstance(entry, SessionMessageEntry)
        assert entry.id == "uf-1"
        assert not hasattr(entry, "future_field")


# ===================================================================
# TestSessionTree
# ===================================================================


class TestSessionTree:
    """Tests for session tree operations."""

    def test_build_tree_empty(self) -> None:
        """build_tree should return None for empty entries."""
        assert build_tree([]) is None

    def test_build_tree_single_entry(self) -> None:
        """build_tree should handle a single root entry."""
        entry = _make_message(id="root", content="root")
        root = build_tree([entry])

        assert root is not None
        assert root.entry.id == "root"
        assert root.children == []

    def test_build_tree_linear_chain(self) -> None:
        """build_tree should build a linear chain correctly."""
        e1 = _make_message(id="e1", timestamp=1.0)
        e2 = _make_message(id="e2", parent_id="e1", timestamp=2.0)
        e3 = _make_message(id="e3", parent_id="e2", timestamp=3.0)

        root = build_tree([e1, e2, e3])

        assert root is not None
        assert root.entry.id == "e1"
        assert len(root.children) == 1
        assert root.children[0].entry.id == "e2"
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].entry.id == "e3"
        assert root.children[0].children[0].children == []

    def test_build_tree_with_branching(self) -> None:
        """build_tree should handle branching (multiple children for a parent)."""
        root = _make_message(id="root", timestamp=1.0)
        branch_a = _make_message(id="a", parent_id="root", timestamp=2.0)
        branch_b = _make_message(id="b", parent_id="root", timestamp=3.0)
        leaf_a = _make_message(id="a1", parent_id="a", timestamp=4.0)

        tree = build_tree([root, branch_a, branch_b, leaf_a])

        assert tree is not None
        assert tree.entry.id == "root"
        assert len(tree.children) == 2
        # Children should be sorted by timestamp
        assert tree.children[0].entry.id == "a"
        assert tree.children[1].entry.id == "b"
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].entry.id == "a1"

    def test_build_tree_children_sorted_by_timestamp(self) -> None:
        """build_tree should sort children by timestamp at every level."""
        root = _make_message(id="root", timestamp=1.0)
        # Insert in reverse timestamp order
        c3 = _make_message(id="c3", parent_id="root", timestamp=30.0)
        c1 = _make_message(id="c1", parent_id="root", timestamp=10.0)
        c2 = _make_message(id="c2", parent_id="root", timestamp=20.0)

        tree = build_tree([root, c3, c1, c2])

        assert tree is not None
        assert [n.entry.id for n in tree.children] == ["c1", "c2", "c3"]

    def test_build_tree_orphan_becomes_root(self) -> None:
        """Entries with parent_id referencing missing entries should be treated as roots."""
        orphan = _make_message(id="orphan", parent_id="missing-parent", timestamp=1.0)
        tree = build_tree([orphan])

        assert tree is not None
        assert tree.entry.id == "orphan"

    def test_get_branches_empty(self) -> None:
        """get_branches should return empty list for empty entries."""
        assert get_branches([]) == []

    def test_get_branches_linear(self) -> None:
        """get_branches should return one branch for a linear chain."""
        e1 = _make_message(id="e1", timestamp=1.0)
        e2 = _make_message(id="e2", parent_id="e1", timestamp=2.0)
        e3 = _make_message(id="e3", parent_id="e2", timestamp=3.0)

        branches = get_branches([e1, e2, e3])
        assert len(branches) == 1
        assert [e.id for e in branches[0]] == ["e1", "e2", "e3"]

    def test_get_branches_finds_all_branches(self) -> None:
        """get_branches should find all leaf-to-root paths."""
        root = _make_message(id="root", timestamp=1.0)
        a = _make_message(id="a", parent_id="root", timestamp=2.0)
        b = _make_message(id="b", parent_id="root", timestamp=3.0)
        a1 = _make_message(id="a1", parent_id="a", timestamp=4.0)

        branches = get_branches([root, a, b, a1])

        # Should have two branches: root->a->a1 and root->b
        assert len(branches) == 2
        branch_ids = sorted([tuple(e.id for e in br) for br in branches])
        assert ("root", "a", "a1") in branch_ids
        assert ("root", "b") in branch_ids

    def test_get_branches_root_to_leaf_order(self) -> None:
        """get_branches should return branches in root-to-leaf order."""
        e1 = _make_message(id="e1", timestamp=1.0)
        e2 = _make_message(id="e2", parent_id="e1", timestamp=2.0)

        branches = get_branches([e1, e2])
        assert len(branches) == 1
        assert branches[0][0].id == "e1"  # root first
        assert branches[0][1].id == "e2"  # leaf last

    def test_walk_to_root(self) -> None:
        """walk_to_root should walk from leaf to root following parent_id."""
        e1 = _make_message(id="e1", timestamp=1.0)
        e2 = _make_message(id="e2", parent_id="e1", timestamp=2.0)
        e3 = _make_message(id="e3", parent_id="e2", timestamp=3.0)

        path = walk_to_root([e1, e2, e3], "e3")

        # Returned in leaf-to-root order
        assert [e.id for e in path] == ["e3", "e2", "e1"]

    def test_walk_to_root_single_entry(self) -> None:
        """walk_to_root from a root entry should return just that entry."""
        e1 = _make_message(id="e1", timestamp=1.0)

        path = walk_to_root([e1], "e1")
        assert [e.id for e in path] == ["e1"]

    def test_walk_to_root_missing_leaf_id(self) -> None:
        """walk_to_root with a nonexistent leaf_id should return empty list."""
        e1 = _make_message(id="e1", timestamp=1.0)

        path = walk_to_root([e1], "nonexistent")
        assert path == []

    def test_walk_to_root_with_cycle_guard(self) -> None:
        """walk_to_root should detect cycles and stop."""
        # Create a cycle: e1 -> e2 -> e3 -> e1
        e1 = _make_message(id="e1", parent_id="e3", timestamp=1.0)
        e2 = _make_message(id="e2", parent_id="e1", timestamp=2.0)
        e3 = _make_message(id="e3", parent_id="e2", timestamp=3.0)

        path = walk_to_root([e1, e2, e3], "e3")

        # Should visit each node at most once (cycle guard)
        ids = [e.id for e in path]
        assert len(ids) == len(set(ids)), "walk_to_root should not revisit nodes"
        assert len(ids) <= 3

    def test_walk_to_root_partial_path(self) -> None:
        """walk_to_root should stop if a parent_id is not in the entry list."""
        e2 = _make_message(id="e2", parent_id="missing", timestamp=2.0)
        e3 = _make_message(id="e3", parent_id="e2", timestamp=3.0)

        path = walk_to_root([e2, e3], "e3")
        assert [e.id for e in path] == ["e3", "e2"]

    def test_find_entry_found(self) -> None:
        """find_entry should return the matching entry."""
        e1 = _make_message(id="target", content="found me")
        e2 = _make_message(id="other", content="not this one")

        result = find_entry([e1, e2], "target")
        assert result is not None
        assert result.id == "target"
        assert isinstance(result, SessionMessageEntry)
        assert result.content == "found me"

    def test_find_entry_not_found(self) -> None:
        """find_entry should return None when no match is found."""
        e1 = _make_message(id="e1")

        result = find_entry([e1], "nonexistent")
        assert result is None

    def test_find_entry_empty_list(self) -> None:
        """find_entry should return None for an empty list."""
        result = find_entry([], "any-id")
        assert result is None


# ===================================================================
# TestSessionManager
# ===================================================================


class TestSessionManager:
    """Tests for the SessionManager high-level API."""

    def test_create_new_session(self, tmp_path: Path) -> None:
        """Creating a SessionManager without session_id should create a new session."""
        mgr = SessionManager(session_dir=tmp_path)

        assert mgr.header is not None
        assert mgr.header.type == "session_header"
        assert mgr.header.id  # non-empty
        assert mgr.entries == []
        assert mgr.leaf_id is None

        # JSONL file should exist on disk
        session_file = tmp_path / f"{mgr.header.id}.jsonl"
        assert session_file.exists()

    def test_append_message_chains_parent_ids(self, tmp_path: Path) -> None:
        """append_message should chain parent_ids through successive appends."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="first")
        assert msg1.parent_id is None  # first entry has no parent
        assert mgr.leaf_id == msg1.id

        msg2 = mgr.append_message(role="assistant", content="second")
        assert msg2.parent_id == msg1.id
        assert mgr.leaf_id == msg2.id

        msg3 = mgr.append_message(role="user", content="third")
        assert msg3.parent_id == msg2.id
        assert mgr.leaf_id == msg3.id

    def test_append_message_persists_to_disk(self, tmp_path: Path) -> None:
        """Appended messages should be recoverable from disk."""
        mgr = SessionManager(session_dir=tmp_path)
        session_id = mgr.header.id

        mgr.append_message(role="user", content="hello")
        mgr.append_message(role="assistant", content="world")

        # Load from disk independently
        session_file = tmp_path / f"{session_id}.jsonl"
        header, entries = load_session(session_file)
        assert header is not None
        assert header.id == session_id
        assert len(entries) == 2
        assert isinstance(entries[0], SessionMessageEntry)
        assert entries[0].content == "hello"
        assert entries[1].content == "world"

    def test_append_model_change(self, tmp_path: Path) -> None:
        """append_model_change should create a ModelChangeEntry."""
        mgr = SessionManager(session_dir=tmp_path)
        msg = mgr.append_message(role="user", content="test")

        mc = mgr.append_model_change(
            prev_model="gpt-4", new_model="claude-3",
            prev_provider="openai", new_provider="anthropic",
        )

        assert isinstance(mc, ModelChangeEntry)
        assert mc.parent_id == msg.id
        assert mc.previous_model == "gpt-4"
        assert mc.new_model == "claude-3"
        assert mgr.leaf_id == mc.id

    def test_append_compaction(self, tmp_path: Path) -> None:
        """append_compaction should create a CompactionEntry."""
        mgr = SessionManager(session_dir=tmp_path)
        msg = mgr.append_message(role="user", content="test")

        comp = mgr.append_compaction(
            summary="Summarized",
            tokens_before=5000,
            tokens_after=1000,
            first_kept_entry_id=msg.id,
        )

        assert isinstance(comp, CompactionEntry)
        assert comp.parent_id == msg.id
        assert comp.summary == "Summarized"
        assert comp.tokens_before == 5000
        assert comp.tokens_after == 1000
        assert comp.first_kept_entry_id == msg.id

    def test_append_thinking_level_change(self, tmp_path: Path) -> None:
        """append_thinking_level_change should create a ThinkingLevelChangeEntry."""
        mgr = SessionManager(session_dir=tmp_path)
        mgr.append_message(role="user", content="test")

        tlc = mgr.append_thinking_level_change(prev="off", new="high")

        assert isinstance(tlc, ThinkingLevelChangeEntry)
        assert tlc.previous_level == "off"
        assert tlc.new_level == "high"

    def test_build_context_empty_session(self, tmp_path: Path) -> None:
        """build_context on an empty session should return empty context."""
        mgr = SessionManager(session_dir=tmp_path)
        ctx = mgr.build_context()

        assert ctx.messages == []
        assert ctx.model_changes == []
        assert ctx.compactions == []
        assert ctx.current_model == ""
        assert ctx.current_thinking_level == "off"

    def test_build_context_returns_messages_in_order(self, tmp_path: Path) -> None:
        """build_context should return messages in root-to-leaf order."""
        mgr = SessionManager(session_dir=tmp_path)

        mgr.append_message(role="user", content="first")
        mgr.append_message(role="assistant", content="second")
        mgr.append_message(role="user", content="third")

        ctx = mgr.build_context()

        assert len(ctx.messages) == 3
        assert ctx.messages[0].content == "first"
        assert ctx.messages[1].content == "second"
        assert ctx.messages[2].content == "third"
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[2].role == "user"

    def test_build_context_includes_model_changes(self, tmp_path: Path) -> None:
        """build_context should track model changes."""
        mgr = SessionManager(session_dir=tmp_path)

        mgr.append_message(role="user", content="hello")
        mgr.append_model_change(prev_model="gpt-4", new_model="claude-3")
        mgr.append_message(role="assistant", content="world")

        ctx = mgr.build_context()

        assert len(ctx.model_changes) == 1
        assert ctx.current_model == "claude-3"
        assert len(ctx.messages) == 2

    def test_build_context_includes_thinking_level_changes(self, tmp_path: Path) -> None:
        """build_context should track thinking level changes."""
        mgr = SessionManager(session_dir=tmp_path)

        mgr.append_message(role="user", content="hello")
        mgr.append_thinking_level_change(prev="off", new="high")
        mgr.append_message(role="assistant", content="thinking hard")

        ctx = mgr.build_context()

        assert ctx.current_thinking_level == "high"

    def test_build_context_with_compaction(self, tmp_path: Path) -> None:
        """build_context should handle compaction by trimming earlier messages."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="old message 1")
        msg2 = mgr.append_message(role="assistant", content="old message 2")
        msg3 = mgr.append_message(role="user", content="kept message")

        # Compaction that keeps from msg3 onward
        mgr.append_compaction(
            summary="Earlier conversation summarized",
            first_kept_entry_id=msg3.id,
        )

        msg4 = mgr.append_message(role="assistant", content="after compaction")

        ctx = mgr.build_context()

        # The compaction should mean msg1 and msg2 are excluded
        # msg3 (first_kept) and msg4 (after compaction) should be present
        contents = [m.content for m in ctx.messages]
        assert "old message 1" not in contents
        assert "old message 2" not in contents
        assert "kept message" in contents
        assert "after compaction" in contents
        assert len(ctx.compactions) == 1

    def test_build_context_with_compaction_no_first_kept(self, tmp_path: Path) -> None:
        """build_context with compaction but no first_kept_entry_id should keep post-compaction entries."""
        mgr = SessionManager(session_dir=tmp_path)

        mgr.append_message(role="user", content="old")
        mgr.append_compaction(summary="Summarized")
        mgr.append_message(role="assistant", content="new")

        ctx = mgr.build_context()

        contents = [m.content for m in ctx.messages]
        assert "new" in contents
        assert len(ctx.compactions) == 1

    def test_navigate_changes_leaf(self, tmp_path: Path) -> None:
        """navigate should move the leaf pointer to the specified entry."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="first")
        msg2 = mgr.append_message(role="assistant", content="second")
        assert mgr.leaf_id == msg2.id

        mgr.navigate(msg1.id)
        assert mgr.leaf_id == msg1.id

        # Appending after navigate creates a branch
        msg3 = mgr.append_message(role="user", content="branch")
        assert msg3.parent_id == msg1.id
        assert mgr.leaf_id == msg3.id

    def test_navigate_nonexistent_entry_raises(self, tmp_path: Path) -> None:
        """navigate should raise ValueError for an unknown entry id."""
        mgr = SessionManager(session_dir=tmp_path)
        mgr.append_message(role="user", content="test")

        with pytest.raises(ValueError, match="not found in session"):
            mgr.navigate("nonexistent-id")

    def test_fork_creates_new_session(self, tmp_path: Path) -> None:
        """fork should create a new session with copied entries up to the fork point."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="hello")
        msg2 = mgr.append_message(role="assistant", content="world")
        msg3 = mgr.append_message(role="user", content="extra")

        # Fork from msg2 -- should include msg1 and msg2 but not msg3
        forked = mgr.fork(msg2.id)

        assert forked.header.id != mgr.header.id
        assert forked.header.parent_session == mgr.header.id
        assert len(forked.entries) == 2

        # The forked entries should have new IDs but preserve content
        contents = [e.content for e in forked.entries if isinstance(e, SessionMessageEntry)]
        assert "hello" in contents
        assert "world" in contents
        assert "extra" not in contents

    def test_fork_preserves_parent_chain(self, tmp_path: Path) -> None:
        """fork should preserve the parent chain structure in the new session."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="a")
        msg2 = mgr.append_message(role="assistant", content="b")
        msg3 = mgr.append_message(role="user", content="c")

        forked = mgr.fork(msg3.id)

        # First entry should have no parent, second should point to first, etc.
        assert forked.entries[0].parent_id is None
        assert forked.entries[1].parent_id == forked.entries[0].id
        assert forked.entries[2].parent_id == forked.entries[1].id

    def test_fork_nonexistent_entry_raises(self, tmp_path: Path) -> None:
        """fork should raise ValueError for an unknown entry id."""
        mgr = SessionManager(session_dir=tmp_path)
        mgr.append_message(role="user", content="test")

        with pytest.raises(ValueError, match="not found in session"):
            mgr.fork("nonexistent-id")

    def test_fork_persists_to_disk(self, tmp_path: Path) -> None:
        """Forked session should be persisted and loadable from disk."""
        mgr = SessionManager(session_dir=tmp_path)
        msg1 = mgr.append_message(role="user", content="hello")
        msg2 = mgr.append_message(role="assistant", content="world")

        forked = mgr.fork(msg2.id)

        # Load the forked session file from disk
        forked_file = tmp_path / f"{forked.header.id}.jsonl"
        assert forked_file.exists()

        header, entries = load_session(forked_file)
        assert header is not None
        assert header.id == forked.header.id
        assert header.parent_session == mgr.header.id
        assert len(entries) == 2

    def test_fork_with_model_change_entries(self, tmp_path: Path) -> None:
        """fork should correctly copy ModelChangeEntry entries."""
        mgr = SessionManager(session_dir=tmp_path)
        msg1 = mgr.append_message(role="user", content="start")
        mc = mgr.append_model_change(prev_model="a", new_model="b")

        forked = mgr.fork(mc.id)
        assert len(forked.entries) == 2
        assert isinstance(forked.entries[0], SessionMessageEntry)
        assert isinstance(forked.entries[1], ModelChangeEntry)
        assert forked.entries[1].new_model == "b"

    def test_load_existing_session(self, tmp_path: Path) -> None:
        """SessionManager should be able to reload an existing session by id."""
        mgr = SessionManager(session_dir=tmp_path)
        session_id = mgr.header.id
        mgr.append_message(role="user", content="persisted")
        mgr.append_message(role="assistant", content="response")

        # Create a new manager that loads the existing session
        reloaded = SessionManager(session_dir=tmp_path, session_id=session_id)

        assert reloaded.header.id == session_id
        assert len(reloaded.entries) == 2
        contents = [
            e.content for e in reloaded.entries if isinstance(e, SessionMessageEntry)
        ]
        assert "persisted" in contents
        assert "response" in contents
        # Leaf should be the last entry
        assert reloaded.leaf_id == reloaded.entries[-1].id

    def test_load_nonexistent_session_raises(self, tmp_path: Path) -> None:
        """Loading a session with a nonexistent id should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No session with id"):
            SessionManager(session_dir=tmp_path, session_id="does-not-exist")

    def test_build_context_after_navigate_uses_branch(self, tmp_path: Path) -> None:
        """build_context after navigate should follow the branch from the new leaf."""
        mgr = SessionManager(session_dir=tmp_path)

        msg1 = mgr.append_message(role="user", content="shared")
        msg2 = mgr.append_message(role="assistant", content="original")

        # Navigate back and create a branch
        mgr.navigate(msg1.id)
        msg3 = mgr.append_message(role="assistant", content="branch response")

        ctx = mgr.build_context()

        contents = [m.content for m in ctx.messages]
        assert "shared" in contents
        assert "branch response" in contents
        assert "original" not in contents

    def test_entries_returns_shallow_copy(self, tmp_path: Path) -> None:
        """The entries property should return a shallow copy, not the internal list."""
        mgr = SessionManager(session_dir=tmp_path)
        mgr.append_message(role="user", content="test")

        entries_a = mgr.entries
        entries_b = mgr.entries
        assert entries_a == entries_b
        assert entries_a is not entries_b  # different list objects

    def test_append_message_with_tool_calls(self, tmp_path: Path) -> None:
        """append_message should handle tool_calls and tool_call_id."""
        mgr = SessionManager(session_dir=tmp_path)

        msg = mgr.append_message(
            role="assistant",
            content="",
            tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "test"}}],
        )
        assert msg.tool_calls == [{"id": "tc1", "type": "function", "function": {"name": "test"}}]

        tool_result = mgr.append_message(
            role="tool",
            content="result data",
            tool_call_id="tc1",
            name="test",
        )
        assert tool_result.tool_call_id == "tc1"
        assert tool_result.name == "test"
        assert tool_result.parent_id == msg.id

    def test_append_message_with_metadata(self, tmp_path: Path) -> None:
        """append_message should pass metadata correctly."""
        mgr = SessionManager(session_dir=tmp_path)

        msg = mgr.append_message(
            role="user",
            content="test",
            metadata={"source": "cli", "tokens": 42},
        )
        assert msg.metadata == {"source": "cli", "tokens": 42}
