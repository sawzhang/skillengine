"""
Terminal UI framework for the skillengine.

Provides components for building interactive terminal interfaces including
text input, multi-line editing, markdown rendering, select lists, overlays,
keybinding management, and autocomplete.
"""

from __future__ import annotations

from skillengine.tui.autocomplete import (
    AutocompleteProvider,
    CombinedAutocomplete,
    CommandAutocomplete,
    FileAutocomplete,
    SlashCommand,
    Suggestion,
)
from skillengine.tui.component import Component
from skillengine.tui.container import Container
from skillengine.tui.editor_widget import EditorWidget
from skillengine.tui.input_widget import InputWidget
from skillengine.tui.keybindings import DEFAULT_KEYBINDINGS, KeybindingsManager
from skillengine.tui.keys import Key, parse_key
from skillengine.tui.markdown_widget import MarkdownWidget
from skillengine.tui.overlay import OverlayManager
from skillengine.tui.renderer import TUIRenderer
from skillengine.tui.select_list import ListItem, SelectList

__all__ = [
    # Core
    "Component",
    "Container",
    "TUIRenderer",
    # Keys
    "Key",
    "parse_key",
    # Widgets
    "InputWidget",
    "EditorWidget",
    "MarkdownWidget",
    "SelectList",
    "ListItem",
    # Overlay
    "OverlayManager",
    # Keybindings
    "KeybindingsManager",
    "DEFAULT_KEYBINDINGS",
    # Autocomplete
    "AutocompleteProvider",
    "CombinedAutocomplete",
    "CommandAutocomplete",
    "FileAutocomplete",
    "SlashCommand",
    "Suggestion",
]
