"""
Abstract base component for TUI widgets.

All renderable elements in the TUI hierarchy inherit from ``Component``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from skillengine.tui.keys import Key


class Component(ABC):
    """
    Base class for TUI components.

    Subclasses must implement :meth:`render` which returns a list of
    pre-styled text lines.  Components track *dirty* state to allow the
    renderer to skip unchanged regions.
    """

    def __init__(self) -> None:
        self._dirty: bool = True
        self._visible: bool = True
        self._focused: bool = False

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """
        Render the component into a list of text lines.

        Each line should be at most *width* visible characters (ANSI
        escape sequences do not count).

        Parameters
        ----------
        width:
            The available horizontal space in columns.

        Returns
        -------
        list[str]
            One string per row.
        """
        ...

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def handle_input(self, key: Key) -> bool:
        """
        Handle a keyboard event.

        Parameters
        ----------
        key:
            The parsed key press.

        Returns
        -------
        bool
            ``True`` if the event was consumed and should not propagate.
        """
        return False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Mark the component as needing a re-render."""
        self._dirty = True

    @property
    def dirty(self) -> bool:
        """Whether the component needs to be re-rendered."""
        return self._dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
        self._dirty = value

    @property
    def visible(self) -> bool:
        """Whether the component is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def focused(self) -> bool:
        """Whether the component currently has input focus."""
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        if self._focused != value:
            self._focused = value
            self._dirty = True
