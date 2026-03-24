"""
Vertical layout container.

Renders child components sequentially from top to bottom, delegating
input to the focused child.
"""

from __future__ import annotations

from skillengine.tui.component import Component
from skillengine.tui.keys import Key


class Container(Component):
    """
    A vertical stack of child :class:`Component` instances.

    Children are rendered in order from top to bottom.  The container
    delegates keyboard input to the currently focused child, or walks
    the children list looking for a consumer.
    """

    def __init__(self, children: list[Component] | None = None) -> None:
        super().__init__()
        self._children: list[Component] = list(children) if children else []
        self._focused_index: int = 0

    # ------------------------------------------------------------------
    # Child management
    # ------------------------------------------------------------------

    @property
    def children(self) -> list[Component]:
        """Return the list of child components."""
        return self._children

    def add(self, child: Component) -> None:
        """Append a child component to the container."""
        self._children.append(child)
        self.invalidate()

    def insert(self, index: int, child: Component) -> None:
        """Insert a child component at *index*."""
        self._children.insert(index, child)
        self.invalidate()

    def remove(self, child: Component) -> None:
        """Remove a child component."""
        self._children.remove(child)
        if self._focused_index >= len(self._children):
            self._focused_index = max(0, len(self._children) - 1)
        self.invalidate()

    def clear(self) -> None:
        """Remove all children."""
        self._children.clear()
        self._focused_index = 0
        self.invalidate()

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    @property
    def focused_index(self) -> int:
        """Index of the child that currently has focus."""
        return self._focused_index

    @focused_index.setter
    def focused_index(self, value: int) -> None:
        if not self._children:
            return
        old = self._focused_index
        self._focused_index = max(0, min(value, len(self._children) - 1))
        if old != self._focused_index:
            if 0 <= old < len(self._children):
                self._children[old].focused = False
            self._children[self._focused_index].focused = True
            self.invalidate()

    def focus_next(self) -> None:
        """Move focus to the next visible child."""
        if not self._children:
            return
        start = self._focused_index
        idx = (start + 1) % len(self._children)
        while idx != start:
            if self._children[idx].visible:
                self.focused_index = idx
                return
            idx = (idx + 1) % len(self._children)

    def focus_prev(self) -> None:
        """Move focus to the previous visible child."""
        if not self._children:
            return
        start = self._focused_index
        idx = (start - 1) % len(self._children)
        while idx != start:
            if self._children[idx].visible:
                self.focused_index = idx
                return
            idx = (idx - 1) % len(self._children)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """Render all visible children top-to-bottom."""
        lines: list[str] = []
        for child in self._children:
            if child.visible:
                lines.extend(child.render(width))
        self._dirty = False
        return lines

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:
        """
        Dispatch input to the focused child.

        If the focused child does not consume the event, iterate over
        remaining children looking for a handler.
        """
        if not self._children:
            return False

        # Try focused child first
        if 0 <= self._focused_index < len(self._children):
            focused = self._children[self._focused_index]
            if focused.visible and focused.handle_input(key):
                return True

        # Fall through to other children
        for i, child in enumerate(self._children):
            if i == self._focused_index:
                continue
            if child.visible and child.handle_input(key):
                return True

        return False

    # ------------------------------------------------------------------
    # Dirty propagation
    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        """A container is dirty if it or any visible child is dirty."""
        if self._dirty:
            return True
        return any(c.dirty for c in self._children if c.visible)

    @dirty.setter
    def dirty(self, value: bool) -> None:
        self._dirty = value

    def invalidate(self) -> None:
        """Mark the container and all children as dirty."""
        self._dirty = True
        for child in self._children:
            child.invalidate()
