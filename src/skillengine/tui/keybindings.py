"""
Keybinding management.

Stores the mapping from logical actions to key sequences, supports
user overrides loaded from a JSON configuration file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillengine.tui.keys import Key

# ---------------------------------------------------------------------------
# Default keybinding map
# ---------------------------------------------------------------------------

DEFAULT_KEYBINDINGS: dict[str, list[str]] = {
    "interrupt": ["ctrl+c"],
    "clear": ["ctrl+l"],
    "exit": ["ctrl+d"],
    "cycle_thinking": ["shift+tab"],
    "cycle_model": ["ctrl+m"],
    "select_model": ["ctrl+shift+m"],
    "follow_up": ["ctrl+f"],
    "new_session": ["ctrl+n"],
    "tree": ["ctrl+t"],
    "fork": ["ctrl+shift+f"],
    "resume": ["ctrl+r"],
}


# ---------------------------------------------------------------------------
# Normalised key descriptor parsing
# ---------------------------------------------------------------------------


def _normalise_key_descriptor(descriptor: str) -> str:
    """
    Normalise a human-readable key descriptor to a canonical form.

    ``"Ctrl+Shift+M"`` -> ``"ctrl+shift+m"``
    """
    parts = [p.strip().lower() for p in descriptor.split("+")]
    # Sort modifiers, keep the base key last
    modifiers = sorted(p for p in parts[:-1])
    base = parts[-1] if parts else ""
    return "+".join(modifiers + [base])


def _key_to_descriptor(key: Key) -> str:
    """
    Convert a parsed :class:`Key` into a canonical descriptor string.

    The result can be compared against normalised binding descriptors.

    Examples
    --------
    >>> _key_to_descriptor(Key(name="ctrl+c", char="c", ctrl=True))
    'ctrl+c'
    >>> _key_to_descriptor(Key(name="tab", shift=True))
    'shift+tab'
    """
    parts: list[str] = []
    if key.ctrl:
        parts.append("ctrl")
    if key.alt:
        parts.append("alt")
    if key.shift:
        parts.append("shift")

    # Determine the base key name.  For ctrl+<letter> combos the name is
    # already ``"ctrl+<letter>"`` so we extract just the base.
    base = key.name
    if "+" in base:
        # e.g. "ctrl+c" -> take the last segment
        base = base.rsplit("+", 1)[-1]

    parts.append(base)

    # Deduplicate and sort modifiers
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        lp = p.lower()
        if lp not in seen:
            seen.add(lp)
            unique.append(lp)
    # Re-sort: modifiers first (alphabetical), base key last
    mods = sorted(p for p in unique[:-1])
    return "+".join(mods + [unique[-1]])


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class KeybindingsManager:
    """
    Manages the mapping from logical action names to key sequences.

    Parameters
    ----------
    user_overrides:
        Optional mapping of action names to key descriptor lists that
        replace the defaults for those actions.
    """

    def __init__(self, user_overrides: dict[str, list[str]] | None = None) -> None:
        self._bindings: dict[str, list[str]] = dict(DEFAULT_KEYBINDINGS)
        if user_overrides:
            self._bindings.update(user_overrides)

        # Pre-normalise all descriptors for fast matching
        self._normalised: dict[str, list[str]] = {
            action: [_normalise_key_descriptor(d) for d in descriptors]
            for action, descriptors in self._bindings.items()
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> KeybindingsManager:
        """
        Load keybindings from a JSON file.

        Search order when *config_path* is ``None``:

        1. ``~/.skillengine/keybindings.json``
        2. Defaults only.

        The JSON file should be a mapping from action names to lists of
        key descriptors, e.g.::

            {
                "interrupt": ["ctrl+c"],
                "exit": ["ctrl+d", "ctrl+q"]
            }
        """
        if config_path is not None:
            path = Path(config_path)
        else:
            path = Path.home() / ".skillengine" / "keybindings.json"

        overrides: dict[str, list[str]] | None = None

        if path.is_file():
            try:
                raw: Any = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    overrides = {}
                    for key, val in raw.items():
                        if isinstance(val, list) and all(isinstance(v, str) for v in val):
                            overrides[key] = val
            except (json.JSONDecodeError, OSError):
                pass  # Silently fall back to defaults

        return cls(user_overrides=overrides)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def matches(self, key: Key | str, action: str) -> bool:
        """
        Test whether *key* matches any binding for *action*.

        Parameters
        ----------
        key:
            Either a :class:`Key` instance or a raw key descriptor string
            (e.g. ``"ctrl+c"``).
        action:
            Logical action name (e.g. ``"interrupt"``).

        Returns
        -------
        bool
            ``True`` if the key matches the action.
        """
        descriptors = self._normalised.get(action)
        if descriptors is None:
            return False

        if isinstance(key, str):
            normalised = _normalise_key_descriptor(key)
        else:
            normalised = _key_to_descriptor(key)

        return normalised in descriptors

    def get_keys(self, action: str) -> list[str]:
        """
        Return all key descriptor strings bound to *action*.

        Returns
        -------
        list[str]
            The descriptors in their original (un-normalised) form.
        """
        return list(self._bindings.get(action, []))

    def actions(self) -> list[str]:
        """Return all registered action names."""
        return list(self._bindings.keys())

    def find_action(self, key: Key | str) -> str | None:
        """
        Find the first action that matches *key*, or ``None``.

        Actions are checked in insertion order.
        """
        for action in self._bindings:
            if self.matches(key, action):
                return action
        return None
