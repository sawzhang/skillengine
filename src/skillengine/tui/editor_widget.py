"""
Multi-line text editor widget.

Supports word wrapping, cursor navigation, text insertion/deletion,
and a configurable submit key binding (default: Ctrl+Enter).
"""

from __future__ import annotations

from collections.abc import Callable

from skillengine.tui.ansi import style
from skillengine.tui.component import Component
from skillengine.tui.keys import Key


class EditorWidget(Component):
    """
    Multi-line text editor with word-wrap.

    Features
    --------
    * Arrow key navigation
    * Home / End to jump within a line
    * Page Up / Page Down scrolling
    * Insert and delete text
    * Word wrap based on available width
    * Ctrl+Enter to submit

    Parameters
    ----------
    on_submit:
        Callback invoked with the current value on submit.
    max_visible_lines:
        Maximum number of display rows shown at once.  ``0`` means unlimited.
    """

    def __init__(
        self,
        on_submit: Callable[[str], None] | None = None,
        max_visible_lines: int = 0,
    ) -> None:
        super().__init__()
        self._on_submit = on_submit
        self._max_visible_lines = max_visible_lines

        # Content is stored as a list of lines (without trailing newlines).
        self._lines: list[str] = [""]
        self._cursor_row: int = 0
        self._cursor_col: int = 0
        self._scroll_offset: int = 0  # first visible wrapped row

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> str:
        """Full text content, with lines joined by ``\\n``."""
        return "\n".join(self._lines)

    @value.setter
    def value(self, text: str) -> None:
        self._lines = text.split("\n") if text else [""]
        self._cursor_row = len(self._lines) - 1
        self._cursor_col = len(self._lines[self._cursor_row])
        self._scroll_offset = 0
        self.invalidate()

    @property
    def on_submit(self) -> Callable[[str], None] | None:
        return self._on_submit

    @on_submit.setter
    def on_submit(self, callback: Callable[[str], None] | None) -> None:
        self._on_submit = callback

    @property
    def line_count(self) -> int:
        """Number of logical lines (before wrapping)."""
        return len(self._lines)

    # ------------------------------------------------------------------
    # Word wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_line(line: str, width: int) -> list[str]:
        """
        Wrap a single logical line into display rows.

        Wraps at word boundaries when possible, otherwise hard-wraps.
        """
        if width <= 0:
            return [line]
        if len(line) <= width:
            return [line]

        rows: list[str] = []
        remaining = line
        while len(remaining) > width:
            # Try to find a space to break at
            break_at = remaining.rfind(" ", 0, width + 1)
            if break_at <= 0:
                # Hard wrap
                break_at = width
            rows.append(remaining[:break_at])
            remaining = remaining[break_at:].lstrip(" ") if break_at < len(remaining) else ""
        if remaining:
            rows.append(remaining)
        return rows or [""]

    def _wrapped_lines(self, width: int) -> list[tuple[int, int, str]]:
        """
        Return all wrapped display rows as ``(logical_row, offset_in_line, text)`` tuples.

        Parameters
        ----------
        width:
            Display width for wrapping.
        """
        result: list[tuple[int, int, str]] = []
        for row_idx, line in enumerate(self._lines):
            wrapped = self._wrap_line(line, width)
            offset = 0
            for segment in wrapped:
                result.append((row_idx, offset, segment))
                offset += len(segment)
                # Account for the space that was consumed in wrapping
                if offset < len(line) and offset > 0:
                    pass  # offset already advanced past the break
        return result

    def _cursor_display_pos(self, wrapped: list[tuple[int, int, str]]) -> int:
        """Return the wrapped-row index where the cursor sits."""
        for i, (row_idx, offset, text) in enumerate(wrapped):
            if row_idx == self._cursor_row:
                end = offset + len(text)
                if offset <= self._cursor_col <= end:
                    return i
                # If cursor is beyond this segment, keep looking at next segments
                # of the same logical row
        # Fallback: last row of the cursor's logical line
        for i in range(len(wrapped) - 1, -1, -1):
            if wrapped[i][0] == self._cursor_row:
                return i
        return 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """Render the editor content with word wrapping and a visible cursor."""
        effective_width = max(1, width)
        wrapped = self._wrapped_lines(effective_width)

        if not wrapped:
            wrapped = [(0, 0, "")]

        # Determine display height
        display_height = len(wrapped)
        if self._max_visible_lines > 0:
            display_height = min(display_height, self._max_visible_lines)

        # Ensure cursor is visible within the scroll window
        cursor_display = self._cursor_display_pos(wrapped)
        if cursor_display < self._scroll_offset:
            self._scroll_offset = cursor_display
        elif cursor_display >= self._scroll_offset + display_height:
            self._scroll_offset = cursor_display - display_height + 1
        self._scroll_offset = max(0, min(self._scroll_offset, len(wrapped) - display_height))

        # Build visible rows
        visible = wrapped[self._scroll_offset : self._scroll_offset + display_height]
        output: list[str] = []

        for row_in_view, (row_idx, offset, text) in enumerate(visible):
            absolute_row = self._scroll_offset + row_in_view
            # Is the cursor on this display row?
            if row_idx == self._cursor_row and self._focused:
                col_in_segment = self._cursor_col - offset
                if 0 <= col_in_segment <= len(text) and absolute_row == cursor_display:
                    before = text[:col_in_segment]
                    after = text[col_in_segment + 1 :] if col_in_segment < len(text) else ""
                    cursor_ch = text[col_in_segment] if col_in_segment < len(text) else " "
                    cursor_display_ch = style(cursor_ch, bold=True, underline=True)
                    output.append(f"{before}{cursor_display_ch}{after}")
                    continue
            output.append(text)

        self._dirty = False
        return output

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:  # noqa: C901
        """Handle keyboard input for the multi-line editor."""
        if not self._focused:
            return False

        name = key.name

        # Submit: Ctrl+Enter
        if name == "enter" and key.ctrl:
            if self._on_submit is not None:
                self._on_submit(self.value)
            return True

        # Newline: Enter
        if name == "enter":
            line = self._lines[self._cursor_row]
            before = line[: self._cursor_col]
            after = line[self._cursor_col :]
            self._lines[self._cursor_row] = before
            self._lines.insert(self._cursor_row + 1, after)
            self._cursor_row += 1
            self._cursor_col = 0
            self.invalidate()
            return True

        # Backspace
        if name == "backspace":
            if self._cursor_col > 0:
                line = self._lines[self._cursor_row]
                self._lines[self._cursor_row] = (
                    line[: self._cursor_col - 1] + line[self._cursor_col :]
                )
                self._cursor_col -= 1
            elif self._cursor_row > 0:
                # Merge with previous line
                prev = self._lines[self._cursor_row - 1]
                self._cursor_col = len(prev)
                self._lines[self._cursor_row - 1] = prev + self._lines[self._cursor_row]
                del self._lines[self._cursor_row]
                self._cursor_row -= 1
            self.invalidate()
            return True

        # Delete
        if name == "delete":
            line = self._lines[self._cursor_row]
            if self._cursor_col < len(line):
                self._lines[self._cursor_row] = (
                    line[: self._cursor_col] + line[self._cursor_col + 1 :]
                )
            elif self._cursor_row < len(self._lines) - 1:
                # Merge with next line
                self._lines[self._cursor_row] = line + self._lines[self._cursor_row + 1]
                del self._lines[self._cursor_row + 1]
            self.invalidate()
            return True

        # Arrow keys
        if name == "left":
            if self._cursor_col > 0:
                self._cursor_col -= 1
            elif self._cursor_row > 0:
                self._cursor_row -= 1
                self._cursor_col = len(self._lines[self._cursor_row])
            self.invalidate()
            return True

        if name == "right":
            line = self._lines[self._cursor_row]
            if self._cursor_col < len(line):
                self._cursor_col += 1
            elif self._cursor_row < len(self._lines) - 1:
                self._cursor_row += 1
                self._cursor_col = 0
            self.invalidate()
            return True

        if name == "up":
            if self._cursor_row > 0:
                self._cursor_row -= 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
                self.invalidate()
            return True

        if name == "down":
            if self._cursor_row < len(self._lines) - 1:
                self._cursor_row += 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
                self.invalidate()
            return True

        # Home / End
        if name == "home" or name == "ctrl+a":
            if self._cursor_col != 0:
                self._cursor_col = 0
                self.invalidate()
            return True

        if name == "end" or name == "ctrl+e":
            end = len(self._lines[self._cursor_row])
            if self._cursor_col != end:
                self._cursor_col = end
                self.invalidate()
            return True

        # Page Up / Page Down
        if name == "page_up":
            page = self._max_visible_lines if self._max_visible_lines > 0 else 20
            self._cursor_row = max(0, self._cursor_row - page)
            self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
            self.invalidate()
            return True

        if name == "page_down":
            page = self._max_visible_lines if self._max_visible_lines > 0 else 20
            self._cursor_row = min(len(self._lines) - 1, self._cursor_row + page)
            self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
            self.invalidate()
            return True

        # Tab: insert spaces
        if name == "tab" and not key.shift:
            indent = "    "
            line = self._lines[self._cursor_row]
            self._lines[self._cursor_row] = (
                line[: self._cursor_col] + indent + line[self._cursor_col :]
            )
            self._cursor_col += len(indent)
            self.invalidate()
            return True

        # Printable character insertion
        if key.char and key.char.isprintable() and not key.ctrl and not key.alt:
            line = self._lines[self._cursor_row]
            self._lines[self._cursor_row] = (
                line[: self._cursor_col] + key.char + line[self._cursor_col :]
            )
            self._cursor_col += 1
            self.invalidate()
            return True

        return False
