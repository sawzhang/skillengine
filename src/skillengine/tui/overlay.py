"""
Modal overlay manager.

Overlays are components that render on top of the main content.  The
:class:`OverlayManager` maintains a stack so that multiple overlays can
be layered (e.g. a confirmation dialog on top of a command palette).
"""

from __future__ import annotations

from skillengine.tui.ansi import style
from skillengine.tui.component import Component
from skillengine.tui.keys import Key


class OverlayManager:
    """
    Stack-based overlay manager.

    Pushed components receive focus and keyboard input before the
    underlying content.  The renderer should query :meth:`compose` to
    merge overlay output on top of the base frame.

    Example
    -------
    >>> mgr = OverlayManager()
    >>> mgr.push(dialog)
    >>> mgr.is_active
    True
    >>> mgr.pop()
    """

    def __init__(self) -> None:
        self._stack: list[Component] = []

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def push(self, component: Component) -> None:
        """
        Push a component onto the overlay stack.

        The new overlay is automatically focused.
        """
        component.focused = True
        component.invalidate()
        self._stack.append(component)

    def pop(self) -> Component | None:
        """
        Remove and return the top overlay, or ``None`` if the stack is empty.

        The popped component loses focus.
        """
        if not self._stack:
            return None
        component = self._stack.pop()
        component.focused = False
        return component

    def clear(self) -> None:
        """Remove all overlays."""
        for comp in self._stack:
            comp.focused = False
        self._stack.clear()

    @property
    def is_active(self) -> bool:
        """``True`` when at least one overlay is on the stack."""
        return bool(self._stack)

    @property
    def top(self) -> Component | None:
        """The topmost overlay component, or ``None``."""
        return self._stack[-1] if self._stack else None

    @property
    def stack(self) -> list[Component]:
        """A copy of the current overlay stack (bottom to top)."""
        return list(self._stack)

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:
        """
        Dispatch input to the topmost overlay.

        Returns ``True`` if an overlay consumed the event, preventing it
        from reaching the underlying content.  If the stack is empty the
        event is *not* consumed.
        """
        if not self._stack:
            return False
        return self._stack[-1].handle_input(key)

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def compose(
        self,
        base_lines: list[str],
        width: int,
        height: int,
    ) -> list[str]:
        """
        Composite overlay output on top of *base_lines*.

        Each overlay is rendered and centered vertically and horizontally
        over the base frame.  A dim border of ``|`` characters is drawn
        around the overlay content.

        Parameters
        ----------
        base_lines:
            The background frame lines (already sized to *height*).
        width:
            Terminal width.
        height:
            Terminal height.

        Returns
        -------
        list[str]
            The composited frame.
        """
        result = list(base_lines)

        # Ensure result is exactly *height* lines
        while len(result) < height:
            result.append("")

        for component in self._stack:
            if not component.visible:
                continue

            overlay_lines = component.render(max(1, width - 4))

            # Center vertically
            overlay_height = len(overlay_lines)
            top = max(0, (height - overlay_height) // 2)

            # Determine overlay content width
            content_width = max(
                (len(self._strip_ansi(line)) for line in overlay_lines),
                default=0,
            )
            box_width = min(content_width + 4, width)  # 2 border + 2 padding
            left_pad = max(0, (width - box_width) // 2)

            # Draw top border
            if top > 0 and top - 1 < height:
                border_top = " " * left_pad + style(
                    "\u250c" + "\u2500" * (box_width - 2) + "\u2510",
                    dim=True,
                )
                result[top - 1] = border_top

            for i, line in enumerate(overlay_lines):
                row = top + i
                if 0 <= row < height:
                    stripped_len = len(self._strip_ansi(line))
                    inner_pad = box_width - 4 - stripped_len
                    padded_line = (
                        " " * left_pad
                        + style("\u2502", dim=True)
                        + " "
                        + line
                        + " " * max(0, inner_pad)
                        + " "
                        + style("\u2502", dim=True)
                    )
                    result[row] = padded_line

            # Draw bottom border
            bottom_row = top + overlay_height
            if 0 <= bottom_row < height:
                border_bottom = " " * left_pad + style(
                    "\u2514" + "\u2500" * (box_width - 2) + "\u2518",
                    dim=True,
                )
                result[bottom_row] = border_bottom

        return result[:height]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences from *text* for length measurement."""
        import re

        return re.sub(r"\033\[[^m]*m", "", text)
