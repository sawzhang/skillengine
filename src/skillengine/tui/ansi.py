"""
ANSI escape sequence utilities for terminal rendering.

Provides color constants, text styling, cursor control, and screen
manipulation primitives used by the TUI framework.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------

ESC = "\033"
CSI = f"{ESC}["
OSC = f"{ESC}]"
RESET = f"{CSI}0m"


# ---------------------------------------------------------------------------
# Standard foreground colors (30-37, 90-97)
# ---------------------------------------------------------------------------


class FG:
    """Standard ANSI foreground colors."""

    BLACK = f"{CSI}30m"
    RED = f"{CSI}31m"
    GREEN = f"{CSI}32m"
    YELLOW = f"{CSI}33m"
    BLUE = f"{CSI}34m"
    MAGENTA = f"{CSI}35m"
    CYAN = f"{CSI}36m"
    WHITE = f"{CSI}37m"
    BRIGHT_BLACK = f"{CSI}90m"
    BRIGHT_RED = f"{CSI}91m"
    BRIGHT_GREEN = f"{CSI}92m"
    BRIGHT_YELLOW = f"{CSI}93m"
    BRIGHT_BLUE = f"{CSI}94m"
    BRIGHT_MAGENTA = f"{CSI}95m"
    BRIGHT_CYAN = f"{CSI}96m"
    BRIGHT_WHITE = f"{CSI}97m"


# ---------------------------------------------------------------------------
# Standard background colors (40-47, 100-107)
# ---------------------------------------------------------------------------


class BG:
    """Standard ANSI background colors."""

    BLACK = f"{CSI}40m"
    RED = f"{CSI}41m"
    GREEN = f"{CSI}42m"
    YELLOW = f"{CSI}43m"
    BLUE = f"{CSI}44m"
    MAGENTA = f"{CSI}45m"
    CYAN = f"{CSI}46m"
    WHITE = f"{CSI}47m"
    BRIGHT_BLACK = f"{CSI}100m"
    BRIGHT_RED = f"{CSI}101m"
    BRIGHT_GREEN = f"{CSI}102m"
    BRIGHT_YELLOW = f"{CSI}103m"
    BRIGHT_BLUE = f"{CSI}104m"
    BRIGHT_MAGENTA = f"{CSI}105m"
    BRIGHT_CYAN = f"{CSI}106m"
    BRIGHT_WHITE = f"{CSI}107m"


# ---------------------------------------------------------------------------
# True-color (24-bit) helpers
# ---------------------------------------------------------------------------


def rgb_fg(r: int, g: int, b: int) -> str:
    """Return an escape sequence for a 24-bit foreground color."""
    return f"{CSI}38;2;{r};{g};{b}m"


def rgb_bg(r: int, g: int, b: int) -> str:
    """Return an escape sequence for a 24-bit background color."""
    return f"{CSI}48;2;{r};{g};{b}m"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string (with or without '#') to an (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_fg(hex_color: str) -> str:
    """Return an escape sequence for a 24-bit foreground color from hex (e.g. '#ff8800')."""
    r, g, b = _hex_to_rgb(hex_color)
    return rgb_fg(r, g, b)


def hex_bg(hex_color: str) -> str:
    """Return an escape sequence for a 24-bit background color from hex (e.g. '#ff8800')."""
    r, g, b = _hex_to_rgb(hex_color)
    return rgb_bg(r, g, b)


# ---------------------------------------------------------------------------
# Text styling
# ---------------------------------------------------------------------------

_STYLE_CODES: dict[str, int] = {
    "bold": 1,
    "dim": 2,
    "italic": 3,
    "underline": 4,
    "strikethrough": 9,
}


def style(
    text: str,
    *,
    fg: str | None = None,
    bg: str | None = None,
    bold: bool = False,
    dim: bool = False,
    italic: bool = False,
    underline: bool = False,
    strikethrough: bool = False,
) -> str:
    """
    Apply ANSI styling to *text*.

    Parameters
    ----------
    text:
        The string to style.
    fg:
        Foreground color -- either an already-formed ANSI sequence (e.g. ``FG.RED``)
        or a hex color string (e.g. ``'#ff0000'``).
    bg:
        Background color -- same format options as *fg*.
    bold, dim, italic, underline, strikethrough:
        Boolean attribute flags.

    Returns
    -------
    str
        The text wrapped in the appropriate ANSI escape sequences with a
        trailing ``RESET``.
    """
    parts: list[str] = []

    # Foreground
    if fg is not None:
        if fg.startswith(ESC) or fg.startswith(CSI):
            parts.append(fg)
        else:
            parts.append(hex_fg(fg))

    # Background
    if bg is not None:
        if bg.startswith(ESC) or bg.startswith(CSI):
            parts.append(bg)
        else:
            parts.append(hex_bg(bg))

    # Attributes
    attrs = {
        "bold": bold,
        "dim": dim,
        "italic": italic,
        "underline": underline,
        "strikethrough": strikethrough,
    }
    for attr_name, enabled in attrs.items():
        if enabled:
            parts.append(f"{CSI}{_STYLE_CODES[attr_name]}m")

    if not parts:
        return text

    prefix = "".join(parts)
    return f"{prefix}{text}{RESET}"


# ---------------------------------------------------------------------------
# Cursor movement
# ---------------------------------------------------------------------------


def cursor_up(n: int = 1) -> str:
    """Move cursor up by *n* rows."""
    return f"{CSI}{n}A"


def cursor_down(n: int = 1) -> str:
    """Move cursor down by *n* rows."""
    return f"{CSI}{n}B"


def cursor_forward(n: int = 1) -> str:
    """Move cursor right by *n* columns."""
    return f"{CSI}{n}C"


def cursor_back(n: int = 1) -> str:
    """Move cursor left by *n* columns."""
    return f"{CSI}{n}D"


def cursor_position(row: int, col: int) -> str:
    """Move cursor to absolute *row*, *col* (1-based)."""
    return f"{CSI}{row};{col}H"


def cursor_save() -> str:
    """Save cursor position (DEC private)."""
    return f"{ESC}7"


def cursor_restore() -> str:
    """Restore cursor position (DEC private)."""
    return f"{ESC}8"


# ---------------------------------------------------------------------------
# Screen / line clearing
# ---------------------------------------------------------------------------


def clear_line() -> str:
    """Erase the entire current line."""
    return f"{CSI}2K"


def clear_screen() -> str:
    """Clear the entire screen and move cursor to top-left."""
    return f"{CSI}2J{CSI}H"


def clear_to_end() -> str:
    """Clear from cursor to end of line."""
    return f"{CSI}0K"


# ---------------------------------------------------------------------------
# Cursor visibility
# ---------------------------------------------------------------------------


def hide_cursor() -> str:
    """Hide the terminal cursor."""
    return f"{CSI}?25l"


def show_cursor() -> str:
    """Show the terminal cursor."""
    return f"{CSI}?25h"


# ---------------------------------------------------------------------------
# Terminal title
# ---------------------------------------------------------------------------


def set_title(title: str) -> str:
    """Set the terminal window title via OSC 2."""
    return f"{OSC}2;{title}{ESC}\\"
