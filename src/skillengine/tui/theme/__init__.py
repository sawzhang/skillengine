"""Theme system for the TUI."""

from __future__ import annotations

from skillengine.tui.theme.defaults import DEFAULT_DARK_THEME, get_default_theme
from skillengine.tui.theme.loader import discover_themes, load_theme
from skillengine.tui.theme.models import ALL_COLOR_KEYS, ThemeColor, ThemeInfo
from skillengine.tui.theme.schema import THEME_SCHEMA, validate_theme

__all__ = [
    "ALL_COLOR_KEYS",
    "DEFAULT_DARK_THEME",
    "THEME_SCHEMA",
    "ThemeColor",
    "ThemeInfo",
    "discover_themes",
    "get_default_theme",
    "load_theme",
    "validate_theme",
]
