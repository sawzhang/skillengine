"""
Arrow-key navigation list widget.

Renders a scrollable, filterable list of items that the user can navigate
with arrow keys and select with Enter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from skillengine.tui.ansi import style
from skillengine.tui.component import Component
from skillengine.tui.keys import Key


@dataclass
class ListItem:
    """
    A single entry in a :class:`SelectList`.

    Attributes
    ----------
    label:
        Display text for the item.
    value:
        Arbitrary value returned when the item is selected.
    description:
        Optional secondary text shown next to the label.
    """

    label: str
    value: object = None
    description: str = ""


class SelectList(Component):
    """
    Interactive list with arrow-key navigation and optional fuzzy filtering.

    Parameters
    ----------
    items:
        Initial list of items.
    on_select:
        Callback invoked with the selected :class:`ListItem` on Enter.
    max_visible:
        Maximum number of items shown at once.  ``0`` means unlimited.
    filterable:
        If ``True``, the user can type to fuzzy-filter the list.
    """

    def __init__(
        self,
        items: list[ListItem] | None = None,
        on_select: Callable[[ListItem], None] | None = None,
        max_visible: int = 0,
        filterable: bool = False,
    ) -> None:
        super().__init__()
        self._items: list[ListItem] = list(items) if items else []
        self._on_select = on_select
        self._max_visible = max_visible
        self._filterable = filterable

        self._selected_index: int = 0
        self._scroll_offset: int = 0
        self._filter_text: str = ""
        self._filtered_items: list[ListItem] | None = None  # None => use _items directly

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[ListItem]:
        """The full (unfiltered) list of items."""
        return self._items

    @items.setter
    def items(self, value: list[ListItem]) -> None:
        self._items = list(value)
        self._filter_text = ""
        self._filtered_items = None
        self._selected_index = 0
        self._scroll_offset = 0
        self.invalidate()

    @property
    def selected_index(self) -> int:
        """Index of the currently highlighted item in the *visible* list."""
        return self._selected_index

    @selected_index.setter
    def selected_index(self, value: int) -> None:
        visible = self._visible_items
        if visible:
            self._selected_index = max(0, min(value, len(visible) - 1))
        else:
            self._selected_index = 0
        self.invalidate()

    @property
    def selected_item(self) -> ListItem | None:
        """The currently highlighted item, or ``None`` if the list is empty."""
        visible = self._visible_items
        if 0 <= self._selected_index < len(visible):
            return visible[self._selected_index]
        return None

    @property
    def on_select(self) -> Callable[[ListItem], None] | None:
        return self._on_select

    @on_select.setter
    def on_select(self, callback: Callable[[ListItem], None] | None) -> None:
        self._on_select = callback

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @property
    def filter_text(self) -> str:
        """Current filter string (only meaningful when *filterable* is ``True``)."""
        return self._filter_text

    @property
    def _visible_items(self) -> list[ListItem]:
        """Items after applying the current filter."""
        if self._filtered_items is not None:
            return self._filtered_items
        return self._items

    def _apply_filter(self) -> None:
        """Recompute the filtered item list from the current filter text."""
        if not self._filter_text:
            self._filtered_items = None
        else:
            query = self._filter_text.lower()
            self._filtered_items = [
                item for item in self._items if _fuzzy_match(query, item.label.lower())
            ]
        self._selected_index = 0
        self._scroll_offset = 0
        self.invalidate()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """Render the list, highlighting the selected item."""
        lines: list[str] = []
        visible = self._visible_items

        # Filter prompt
        if self._filterable and self._focused:
            prefix = style("Filter: ", bold=True)
            lines.append(f"{prefix}{self._filter_text}")

        if not visible:
            lines.append(style("  (no items)", dim=True))
            self._dirty = False
            return lines

        # Compute display window
        display_count = len(visible)
        if self._max_visible > 0:
            display_count = min(display_count, self._max_visible)

        # Adjust scroll offset to keep selection visible
        if self._selected_index < self._scroll_offset:
            self._scroll_offset = self._selected_index
        elif self._selected_index >= self._scroll_offset + display_count:
            self._scroll_offset = self._selected_index - display_count + 1
        self._scroll_offset = max(0, min(self._scroll_offset, len(visible) - display_count))

        window = visible[self._scroll_offset : self._scroll_offset + display_count]

        for i, item in enumerate(window):
            abs_idx = self._scroll_offset + i
            is_selected = abs_idx == self._selected_index

            marker = ">" if is_selected else " "
            label = item.label
            desc = f"  {item.description}" if item.description else ""

            # Truncate to fit width
            content = f" {marker} {label}{desc}"
            if len(content) > width:
                content = content[: width - 1] + "\u2026"

            if is_selected:
                lines.append(style(content, bold=True, underline=True))
            else:
                lines.append(content)

        # Scroll indicators
        if self._scroll_offset > 0:
            lines.insert(
                len(lines) - display_count if self._filterable else 0,
                style("  \u25b2 more above", dim=True),
            )
        if self._scroll_offset + display_count < len(visible):
            lines.append(style("  \u25bc more below", dim=True))

        self._dirty = False
        return lines

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:
        """Handle navigation and selection input."""
        if not self._focused:
            return False

        name = key.name
        visible = self._visible_items

        # Navigation
        if name == "up":
            if self._selected_index > 0:
                self._selected_index -= 1
                self.invalidate()
            return True

        if name == "down":
            if self._selected_index < len(visible) - 1:
                self._selected_index += 1
                self.invalidate()
            return True

        if name == "home":
            self._selected_index = 0
            self.invalidate()
            return True

        if name == "end":
            self._selected_index = max(0, len(visible) - 1)
            self.invalidate()
            return True

        if name == "page_up":
            page = self._max_visible if self._max_visible > 0 else 10
            self._selected_index = max(0, self._selected_index - page)
            self.invalidate()
            return True

        if name == "page_down":
            page = self._max_visible if self._max_visible > 0 else 10
            self._selected_index = min(len(visible) - 1, self._selected_index + page)
            self.invalidate()
            return True

        # Selection
        if name == "enter":
            item = self.selected_item
            if item is not None and self._on_select is not None:
                self._on_select(item)
            return True

        # Filter text input
        if self._filterable:
            if name == "backspace":
                if self._filter_text:
                    self._filter_text = self._filter_text[:-1]
                    self._apply_filter()
                return True

            if key.char and key.char.isprintable() and not key.ctrl and not key.alt:
                self._filter_text += key.char
                self._apply_filter()
                return True

            # Ctrl+U clears the filter
            if name == "ctrl+u":
                self._filter_text = ""
                self._apply_filter()
                return True

        return False


# ---------------------------------------------------------------------------
# Fuzzy matching helper
# ---------------------------------------------------------------------------


def _fuzzy_match(query: str, text: str) -> bool:
    """
    Simple fuzzy match: every character in *query* must appear in *text*
    in order, but not necessarily contiguously.

    >>> _fuzzy_match("abc", "aXbXc")
    True
    >>> _fuzzy_match("abc", "bac")
    False
    """
    it = iter(text)
    return all(ch in it for ch in query)
