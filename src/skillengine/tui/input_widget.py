"""
Single-line text input widget.

Supports cursor movement, text editing, command history, and a kill ring
(Ctrl+K / Ctrl+Y).
"""

from __future__ import annotations

from collections.abc import Callable

from skillengine.tui.ansi import style
from skillengine.tui.component import Component
from skillengine.tui.keys import Key


class InputWidget(Component):
    """
    Single-line text input with readline-style key bindings.

    Features
    --------
    * Left/Right, Home/End cursor navigation
    * Insert and delete characters
    * History navigation with Up/Down arrows
    * Kill ring: Ctrl+K kills to end of line, Ctrl+Y yanks it back
    * Configurable prompt prefix and submit callback

    Parameters
    ----------
    prompt:
        Text displayed before the editable area (e.g. ``"> "``).
    on_submit:
        Callback invoked with the current value when Enter is pressed.
    """

    def __init__(
        self,
        prompt: str = "> ",
        on_submit: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._prompt = prompt
        self._on_submit = on_submit

        # Editor state
        self._buffer: list[str] = []
        self._cursor: int = 0

        # History
        self._history: list[str] = []
        self._history_index: int = -1
        self._saved_buffer: str = ""  # buffer contents before history navigation

        # Kill ring
        self._kill_ring: list[str] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> str:
        """Current text content of the input."""
        return "".join(self._buffer)

    @value.setter
    def value(self, text: str) -> None:
        self._buffer = list(text)
        self._cursor = len(self._buffer)
        self.invalidate()

    @property
    def prompt(self) -> str:
        """The prompt prefix displayed before the input area."""
        return self._prompt

    @prompt.setter
    def prompt(self, text: str) -> None:
        self._prompt = text
        self.invalidate()

    @property
    def on_submit(self) -> Callable[[str], None] | None:
        return self._on_submit

    @on_submit.setter
    def on_submit(self, callback: Callable[[str], None] | None) -> None:
        self._on_submit = callback

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_history(self, entry: str) -> None:
        """Add an entry to the command history."""
        if entry and (not self._history or self._history[-1] != entry):
            self._history.append(entry)
        self._history_index = -1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """
        Render the prompt and input buffer with a visible cursor.

        The cursor is indicated by a reverse-video character.
        """
        prefix = self._prompt
        text = self.value
        cursor = self._cursor

        # Compute visible window if text exceeds width
        available = max(1, width - len(prefix))
        start = 0
        if cursor > available - 1:
            start = cursor - available + 1
        visible = text[start : start + available]
        cursor_in_view = cursor - start

        # Build the line with cursor highlighting
        before = visible[:cursor_in_view]
        after = visible[cursor_in_view + 1 :] if cursor_in_view < len(visible) else ""
        cursor_char = visible[cursor_in_view] if cursor_in_view < len(visible) else " "

        if self._focused:
            cursor_display = style(cursor_char, bold=True, underline=True)
        else:
            cursor_display = cursor_char

        line = f"{prefix}{before}{cursor_display}{after}"

        self._dirty = False
        return [line]

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:  # noqa: C901 (complex but cohesive)
        """Handle keyboard input for the single-line editor."""
        if not self._focused:
            return False

        name = key.name

        # Submit
        if name == "enter":
            text = self.value
            self.add_history(text)
            if self._on_submit is not None:
                self._on_submit(text)
            return True

        # Cursor movement
        if name == "left" and not key.ctrl:
            if self._cursor > 0:
                self._cursor -= 1
                self.invalidate()
            return True

        if name == "right" and not key.ctrl:
            if self._cursor < len(self._buffer):
                self._cursor += 1
                self.invalidate()
            return True

        if name == "home" or (name == "ctrl+a"):
            if self._cursor != 0:
                self._cursor = 0
                self.invalidate()
            return True

        if name == "end" or (name == "ctrl+e"):
            end = len(self._buffer)
            if self._cursor != end:
                self._cursor = end
                self.invalidate()
            return True

        # Word movement (Ctrl+Left / Ctrl+Right)
        if name == "left" and key.ctrl:
            self._cursor = self._word_boundary_left()
            self.invalidate()
            return True

        if name == "right" and key.ctrl:
            self._cursor = self._word_boundary_right()
            self.invalidate()
            return True

        # Backspace
        if name == "backspace":
            if self._cursor > 0:
                self._cursor -= 1
                del self._buffer[self._cursor]
                self.invalidate()
            return True

        # Delete
        if name == "delete" or name == "ctrl+d":
            if self._cursor < len(self._buffer):
                del self._buffer[self._cursor]
                self.invalidate()
            return True

        # Kill to end of line (Ctrl+K)
        if name == "ctrl+k":
            killed = "".join(self._buffer[self._cursor :])
            if killed:
                self._kill_ring.append(killed)
            self._buffer = self._buffer[: self._cursor]
            self.invalidate()
            return True

        # Kill to start of line (Ctrl+U)
        if name == "ctrl+u":
            killed = "".join(self._buffer[: self._cursor])
            if killed:
                self._kill_ring.append(killed)
            self._buffer = self._buffer[self._cursor :]
            self._cursor = 0
            self.invalidate()
            return True

        # Yank (Ctrl+Y)
        if name == "ctrl+y":
            if self._kill_ring:
                text = self._kill_ring[-1]
                for ch in text:
                    self._buffer.insert(self._cursor, ch)
                    self._cursor += 1
                self.invalidate()
            return True

        # Clear line (Ctrl+W - kill word backward)
        if name == "ctrl+w":
            boundary = self._word_boundary_left()
            killed = "".join(self._buffer[boundary : self._cursor])
            if killed:
                self._kill_ring.append(killed)
            self._buffer = self._buffer[:boundary] + self._buffer[self._cursor :]
            self._cursor = boundary
            self.invalidate()
            return True

        # History navigation
        if name == "up":
            self._history_prev()
            return True

        if name == "down":
            self._history_next()
            return True

        # Printable character insertion
        if key.char and key.char.isprintable() and not key.ctrl and not key.alt:
            self._buffer.insert(self._cursor, key.char)
            self._cursor += 1
            self.invalidate()
            return True

        return False

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _history_prev(self) -> None:
        """Navigate to the previous history entry."""
        if not self._history:
            return
        if self._history_index == -1:
            self._saved_buffer = self.value
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return

        entry = self._history[self._history_index]
        self._buffer = list(entry)
        self._cursor = len(self._buffer)
        self.invalidate()

    def _history_next(self) -> None:
        """Navigate to the next history entry, or restore saved buffer."""
        if self._history_index == -1:
            return

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            entry = self._history[self._history_index]
        else:
            self._history_index = -1
            entry = self._saved_buffer

        self._buffer = list(entry)
        self._cursor = len(self._buffer)
        self.invalidate()

    # ------------------------------------------------------------------
    # Word boundary helpers
    # ------------------------------------------------------------------

    def _word_boundary_left(self) -> int:
        """Find the start of the word to the left of the cursor."""
        pos = self._cursor - 1
        # Skip whitespace
        while pos >= 0 and not self._buffer[pos].isalnum():
            pos -= 1
        # Skip word characters
        while pos >= 0 and self._buffer[pos].isalnum():
            pos -= 1
        return pos + 1

    def _word_boundary_right(self) -> int:
        """Find the end of the word to the right of the cursor."""
        pos = self._cursor
        length = len(self._buffer)
        # Skip whitespace
        while pos < length and not self._buffer[pos].isalnum():
            pos += 1
        # Skip word characters
        while pos < length and self._buffer[pos].isalnum():
            pos += 1
        return pos
