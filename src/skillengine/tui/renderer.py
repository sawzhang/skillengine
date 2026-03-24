"""
Differential rendering engine.

``TUIRenderer`` tracks the previously written frame and only rewrites rows
that changed, using CSI 2026 synchronized output markers to avoid visible
tearing on modern terminals.
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import TextIO

from skillengine.tui.ansi import (
    clear_line,
    clear_screen,
    cursor_position,
    hide_cursor,
    show_cursor,
)
from skillengine.tui.component import Component

# ---------------------------------------------------------------------------
# Synchronized output markers (DEC private mode 2026)
# ---------------------------------------------------------------------------

_SYNC_START = "\033[?2026h"
_SYNC_END = "\033[?2026l"


class TUIRenderer:
    """
    Differential terminal renderer.

    Keeps a copy of the last frame (list of strings, one per row) and on
    each :meth:`render` call only rewrites the rows that differ.  A full
    redraw is forced when the terminal dimensions change.

    Parameters
    ----------
    output:
        Writable text stream, defaults to ``sys.stdout``.
    """

    def __init__(self, output: TextIO | None = None) -> None:
        self._output: TextIO = output or sys.stdout
        self._previous_lines: list[str] = []
        self._prev_width: int = 0
        self._prev_height: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def previous_lines(self) -> list[str]:
        """The last frame that was written to the terminal."""
        return list(self._previous_lines)

    def render(
        self,
        components: list[Component],
        width: int,
        height: int,
    ) -> None:
        """
        Render visible components into the terminal area.

        Only changed rows are rewritten.  A full redraw is triggered when
        *width* or *height* differs from the previous call.

        Parameters
        ----------
        components:
            Ordered list of components to render top-to-bottom.
        width:
            Terminal width in columns.
        height:
            Terminal height in rows.
        """
        # Collect new lines from all visible components
        new_lines: list[str] = []
        for comp in components:
            if comp.visible:
                new_lines.extend(comp.render(width))

        # Pad or truncate to fit height
        if len(new_lines) < height:
            new_lines.extend([""] * (height - len(new_lines)))
        elif len(new_lines) > height:
            new_lines = new_lines[:height]

        # Decide between full and differential redraw
        size_changed = width != self._prev_width or height != self._prev_height
        if size_changed or not self._previous_lines:
            self._full_render(new_lines, width, height)
        else:
            updates = self._diff_render(self._previous_lines, new_lines)
            if updates:
                self._apply_updates(updates)

        # Save state
        self._previous_lines = new_lines
        self._prev_width = width
        self._prev_height = height

        # Mark all components as clean
        for comp in components:
            comp.dirty = False

    def full_redraw(
        self,
        components: list[Component],
        width: int,
        height: int,
    ) -> None:
        """Force a complete redraw regardless of diff state."""
        self._previous_lines = []
        self.render(components, width, height)

    def clear(self) -> None:
        """Clear the screen and reset internal state."""
        self._write(clear_screen())
        self._previous_lines = []
        self._prev_width = 0
        self._prev_height = 0

    # ------------------------------------------------------------------
    # Diff engine
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_render(
        old_lines: list[str],
        new_lines: list[str],
    ) -> list[tuple[int, str]]:
        """
        Compare two frames and return a list of ``(row, text)`` updates.

        Parameters
        ----------
        old_lines:
            Previous frame lines.
        new_lines:
            Current frame lines.

        Returns
        -------
        list[tuple[int, str]]
            Pairs of ``(1-based row number, line text)`` for each changed row.
        """
        max_len = max(len(old_lines), len(new_lines))
        updates: list[tuple[int, str]] = []
        for i in range(max_len):
            old = old_lines[i] if i < len(old_lines) else ""
            new = new_lines[i] if i < len(new_lines) else ""
            if old != new:
                updates.append((i + 1, new))  # 1-based row
        return updates

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _full_render(self, lines: list[str], width: int, height: int) -> None:
        """Write all lines to the terminal, clearing the screen first."""
        buf = StringIO()
        buf.write(_SYNC_START)
        buf.write(hide_cursor())
        buf.write(clear_screen())

        for row_idx, line in enumerate(lines):
            buf.write(cursor_position(row_idx + 1, 1))
            buf.write(line)

        buf.write(show_cursor())
        buf.write(_SYNC_END)
        self._write(buf.getvalue())

    def _apply_updates(self, updates: list[tuple[int, str]]) -> None:
        """Write only the changed rows."""
        buf = StringIO()
        buf.write(_SYNC_START)
        buf.write(hide_cursor())

        for row, text in updates:
            buf.write(cursor_position(row, 1))
            buf.write(clear_line())
            buf.write(text)

        buf.write(show_cursor())
        buf.write(_SYNC_END)
        self._write(buf.getvalue())

    def _write(self, data: str) -> None:
        """Write data to the output stream and flush."""
        self._output.write(data)
        self._output.flush()
