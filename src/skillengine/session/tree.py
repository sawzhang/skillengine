"""
Tree operations for session entries.

Session entries form a tree via parent_id pointers. This module provides
utilities for building the tree structure, walking paths, and extracting
branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skillengine.session.models import SessionEntry


@dataclass
class SessionTreeNode:
    """A node in the session tree with its children."""

    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)


def build_tree(entries: list[SessionEntry]) -> SessionTreeNode | None:
    """
    Build a tree from a flat list of session entries.

    Returns the root node, or ``None`` if the list is empty. If there are
    multiple roots (orphan entries with no valid parent), only the first root
    is returned; remaining orphans are attached as additional roots' siblings
    under a synthetic walk. For a multi-root result use :func:`get_branches`.

    Entries are linked via their ``id`` / ``parent_id`` fields.  An entry
    whose ``parent_id`` is ``None`` (or references a missing id) is treated
    as a root.
    """
    if not entries:
        return None

    # Build lookup of id -> node
    node_map: dict[str, SessionTreeNode] = {}
    for entry in entries:
        node_map[entry.id] = SessionTreeNode(entry=entry)

    roots: list[SessionTreeNode] = []

    for entry in entries:
        node = node_map[entry.id]
        if entry.parent_id is None or entry.parent_id not in node_map:
            roots.append(node)
        else:
            parent_node = node_map[entry.parent_id]
            parent_node.children.append(node)

    # Sort children by timestamp at every level (iterative to avoid stack
    # overflow on deep trees, matching the pi-mono pattern)
    stack: list[SessionTreeNode] = list(roots)
    while stack:
        node = stack.pop()
        node.children.sort(key=lambda n: n.entry.timestamp)
        stack.extend(node.children)

    return roots[0] if roots else None


def get_branches(entries: list[SessionEntry]) -> list[list[SessionEntry]]:
    """
    Get all leaf-to-root paths (branches) in the entry tree.

    Each returned list is ordered from root to leaf (i.e. chronological
    order along that branch).
    """
    if not entries:
        return []

    # Build child map
    children_map: dict[str, list[str]] = {}
    entry_map: dict[str, SessionEntry] = {}
    for entry in entries:
        entry_map[entry.id] = entry
        children_map.setdefault(entry.id, [])
        if entry.parent_id is not None and entry.parent_id in entry_map:
            children_map.setdefault(entry.parent_id, []).append(entry.id)

    # Rebuild child map after all entries are indexed (handles forward refs)
    children_map = {eid: [] for eid in entry_map}
    for entry in entries:
        if entry.parent_id is not None and entry.parent_id in entry_map:
            children_map[entry.parent_id].append(entry.id)

    # Find leaves (entries with no children)
    leaves = [eid for eid, kids in children_map.items() if not kids]

    branches: list[list[SessionEntry]] = []
    for leaf_id in leaves:
        path = walk_to_root(entries, leaf_id)
        # walk_to_root returns leaf-to-root; reverse for root-to-leaf
        path.reverse()
        branches.append(path)

    return branches


def walk_to_root(
    entries: list[SessionEntry],
    leaf_id: str,
) -> list[SessionEntry]:
    """
    Walk from ``leaf_id`` to the root, following ``parent_id`` links.

    Returns a list ordered from **leaf to root**.  The caller can reverse
    it if root-to-leaf ordering is needed (e.g. for building LLM context).
    """
    entry_map: dict[str, SessionEntry] = {e.id: e for e in entries}

    path: list[SessionEntry] = []
    visited: set[str] = set()
    current_id: str | None = leaf_id

    while current_id is not None:
        if current_id in visited:
            break  # Guard against cycles
        visited.add(current_id)

        entry = entry_map.get(current_id)
        if entry is None:
            break
        path.append(entry)
        current_id = entry.parent_id

    return path


def find_entry(
    entries: list[SessionEntry],
    entry_id: str,
) -> SessionEntry | None:
    """Find an entry by id, or return ``None`` if not found."""
    for entry in entries:
        if entry.id == entry_id:
            return entry
    return None
