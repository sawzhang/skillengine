"""
Key parsing for terminal input.

Translates raw bytes read from stdin into structured ``Key`` objects that
the rest of the TUI framework can dispatch on.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Key data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """
    Parsed representation of a single key press.

    Attributes
    ----------
    name:
        Symbolic name for special keys (e.g. ``'enter'``, ``'up'``).
        For plain printable characters this equals *char*.
    char:
        The literal character, if printable.  Empty string otherwise.
    ctrl:
        ``True`` when Ctrl was held.
    alt:
        ``True`` when Alt (Meta/Option) was held.
    shift:
        ``True`` when Shift was held (only detectable for certain keys).
    """

    name: str
    char: str = ""
    ctrl: bool = False
    alt: bool = False
    shift: bool = False


# ---------------------------------------------------------------------------
# Common key constants
# ---------------------------------------------------------------------------

KEY_ENTER = Key(name="enter", char="\r")
KEY_TAB = Key(name="tab", char="\t")
KEY_ESCAPE = Key(name="escape")
KEY_BACKSPACE = Key(name="backspace")
KEY_DELETE = Key(name="delete")
KEY_INSERT = Key(name="insert")

KEY_UP = Key(name="up")
KEY_DOWN = Key(name="down")
KEY_LEFT = Key(name="left")
KEY_RIGHT = Key(name="right")

KEY_HOME = Key(name="home")
KEY_END = Key(name="end")
KEY_PAGE_UP = Key(name="page_up")
KEY_PAGE_DOWN = Key(name="page_down")

KEY_F1 = Key(name="f1")
KEY_F2 = Key(name="f2")
KEY_F3 = Key(name="f3")
KEY_F4 = Key(name="f4")
KEY_F5 = Key(name="f5")
KEY_F6 = Key(name="f6")
KEY_F7 = Key(name="f7")
KEY_F8 = Key(name="f8")
KEY_F9 = Key(name="f9")
KEY_F10 = Key(name="f10")
KEY_F11 = Key(name="f11")
KEY_F12 = Key(name="f12")

KEY_SPACE = Key(name="space", char=" ")


# ---------------------------------------------------------------------------
# CSI (Control Sequence Introducer) lookup tables
# ---------------------------------------------------------------------------

_CSI_SIMPLE: dict[bytes, Key] = {
    b"A": KEY_UP,
    b"B": KEY_DOWN,
    b"C": KEY_RIGHT,
    b"D": KEY_LEFT,
    b"H": KEY_HOME,
    b"F": KEY_END,
    b"Z": Key(name="tab", char="\t", shift=True),  # Shift+Tab
}

# Sequences of the form CSI <number> ~ (e.g. \x1b[3~  for delete)
_CSI_TILDE: dict[int, Key] = {
    1: KEY_HOME,
    2: KEY_INSERT,
    3: KEY_DELETE,
    4: KEY_END,
    5: KEY_PAGE_UP,
    6: KEY_PAGE_DOWN,
    11: KEY_F1,
    12: KEY_F2,
    13: KEY_F3,
    14: KEY_F4,
    15: KEY_F5,
    17: KEY_F6,
    18: KEY_F7,
    19: KEY_F8,
    20: KEY_F9,
    21: KEY_F10,
    23: KEY_F11,
    24: KEY_F12,
}

# SS3 sequences (ESC O <letter>)
_SS3: dict[bytes, Key] = {
    b"P": KEY_F1,
    b"Q": KEY_F2,
    b"R": KEY_F3,
    b"S": KEY_F4,
    b"H": KEY_HOME,
    b"F": KEY_END,
}


# ---------------------------------------------------------------------------
# Modifier bit handling (xterm-style ;N suffixes)
# ---------------------------------------------------------------------------


def _modifier_flags(code: int) -> tuple[bool, bool, bool]:
    """
    Decode an xterm modifier code into ``(shift, alt, ctrl)`` booleans.

    The modifier value is 1-based: ``value = 1 + (shift) + 2*(alt) + 4*(ctrl)``.
    """
    code -= 1  # remove the base offset
    shift = bool(code & 1)
    alt = bool(code & 2)
    ctrl = bool(code & 4)
    return shift, alt, ctrl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_key(data: bytes) -> Key:
    """
    Parse raw terminal input bytes into a ``Key`` object.

    Handles:
    * Printable ASCII and UTF-8 characters
    * Ctrl+letter combinations (bytes 0x01-0x1a)
    * Alt+letter (ESC followed by a character)
    * CSI sequences (arrow keys, function keys, home/end, etc.)
    * SS3 sequences (alternate function key encoding)
    * xterm-style modifier suffixes (e.g. ``CSI 1;5C`` for Ctrl+Right)

    Parameters
    ----------
    data:
        Raw bytes read from the terminal.

    Returns
    -------
    Key
        Structured representation of the key press.
    """
    if not data:
        return Key(name="unknown")

    # -----------------------------------------------------------------------
    # ESC-prefixed sequences
    # -----------------------------------------------------------------------
    if data[0:1] == b"\x1b":
        # Bare ESC
        if len(data) == 1:
            return KEY_ESCAPE

        second = data[1:2]

        # CSI sequence: ESC [
        if second == b"[":
            return _parse_csi(data[2:])

        # SS3 sequence: ESC O
        if second == b"O":
            tail = data[2:3]
            if tail in _SS3:
                return _SS3[tail]
            return Key(name="unknown")

        # Alt+character: ESC followed by printable byte
        if len(data) == 2:
            ch = chr(data[1])
            if ch.isprintable():
                return Key(name=f"alt+{ch}", char=ch, alt=True)
            # Alt + ctrl-letter
            if 1 <= data[1] <= 26:
                letter = chr(data[1] + 96)  # 1 -> 'a'
                return Key(name=f"ctrl+{letter}", char=letter, ctrl=True, alt=True)

        return Key(name="unknown")

    # -----------------------------------------------------------------------
    # Control characters (0x00-0x1f)
    # -----------------------------------------------------------------------
    byte = data[0]

    if byte == 0x0D or byte == 0x0A:  # CR or LF
        return KEY_ENTER

    if byte == 0x09:  # TAB
        return KEY_TAB

    if byte == 0x7F or byte == 0x08:  # DEL or BS
        return KEY_BACKSPACE

    if byte == 0x00:  # Ctrl+Space / Ctrl+@
        return Key(name="ctrl+space", char=" ", ctrl=True)

    if 1 <= byte <= 26:
        letter = chr(byte + 96)  # 1 -> 'a', 2 -> 'b', ...
        return Key(name=f"ctrl+{letter}", char=letter, ctrl=True)

    if byte == 0x1C:
        return Key(name="ctrl+\\", char="\\", ctrl=True)
    if byte == 0x1D:
        return Key(name="ctrl+]", char="]", ctrl=True)
    if byte == 0x1E:
        return Key(name="ctrl+^", char="^", ctrl=True)
    if byte == 0x1F:
        return Key(name="ctrl+_", char="_", ctrl=True)

    # -----------------------------------------------------------------------
    # Printable characters (possibly multi-byte UTF-8)
    # -----------------------------------------------------------------------
    try:
        ch = data.decode("utf-8")
    except UnicodeDecodeError:
        return Key(name="unknown")

    if len(ch) == 1 and ch.isprintable():
        if ch == " ":
            return KEY_SPACE
        return Key(name=ch, char=ch)

    return Key(name="unknown")


# ---------------------------------------------------------------------------
# Internal CSI parser
# ---------------------------------------------------------------------------


def _parse_csi(payload: bytes) -> Key:
    """
    Parse the bytes *after* ``ESC [`` in a CSI sequence.

    Supports:
    * Simple final-byte sequences (e.g. ``A`` for Up)
    * ``<number> ~`` sequences (e.g. ``3~`` for Delete)
    * ``1;<mod> <letter>`` modifier sequences (e.g. ``1;5C`` for Ctrl+Right)
    * ``<number>;<mod> ~`` modifier+tilde sequences
    """
    if not payload:
        return Key(name="unknown")

    # Simple single-byte final character
    if len(payload) == 1 and payload in _CSI_SIMPLE:
        return _CSI_SIMPLE[payload]

    # Decode payload as ASCII for numeric parsing
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        return Key(name="unknown")

    # Tilde sequences: <num>~ or <num>;<mod>~
    if text.endswith("~"):
        inner = text[:-1]
        parts = inner.split(";")
        num = _safe_int(parts[0])
        if num is None:
            return Key(name="unknown")

        base_key = _CSI_TILDE.get(num)
        if base_key is None:
            return Key(name="unknown")

        if len(parts) == 2:
            mod = _safe_int(parts[1])
            if mod is not None:
                shift, alt, ctrl = _modifier_flags(mod)
                return Key(
                    name=base_key.name,
                    char=base_key.char,
                    ctrl=ctrl,
                    alt=alt,
                    shift=shift,
                )

        return base_key

    # Modifier sequences: <num>;<mod><letter>  (e.g. "1;5C" = Ctrl+Right)
    final_char = text[-1:]
    if final_char.isalpha() and ";" in text:
        inner = text[:-1]
        parts = inner.split(";")
        if len(parts) == 2:
            mod = _safe_int(parts[1])
            if mod is not None:
                shift, alt, ctrl = _modifier_flags(mod)
                base = _CSI_SIMPLE.get(final_char.encode("ascii"))
                if base is not None:
                    return Key(
                        name=base.name,
                        char=base.char,
                        ctrl=ctrl,
                        alt=alt,
                        shift=shift,
                    )
        return Key(name="unknown")

    # Single final letter (without preceding digits) that we already checked
    if len(text) == 1 and text.encode("ascii") in _CSI_SIMPLE:
        return _CSI_SIMPLE[text.encode("ascii")]

    return Key(name="unknown")


def _safe_int(s: str) -> int | None:
    """Return ``int(s)`` or ``None`` if *s* is not a valid integer."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return None
