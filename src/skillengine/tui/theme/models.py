"""
Theme data models.

Defines the colour keys and metadata structures that themes must provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Exhaustive list of colour keys
# ---------------------------------------------------------------------------

# Each key maps to a hex colour string (e.g. ``"#1e1e2e"``).
# The keys are grouped by UI domain for readability.

CORE_UI_KEYS: list[str] = [
    "primary",
    "secondary",
    "accent",
    "background",
    "surface",
    "surface_variant",
    "border",
    "border_focused",
    "text",
    "text_muted",
    "text_dim",
    "text_inverse",
    "error",
    "warning",
    "success",
    "info",
]

MESSAGE_KEYS: list[str] = [
    "message_user_bg",
    "message_user_fg",
    "message_assistant_bg",
    "message_assistant_fg",
    "message_system_bg",
    "message_system_fg",
    "message_tool_bg",
    "message_tool_fg",
]

MARKDOWN_KEYS: list[str] = [
    "md_heading1",
    "md_heading2",
    "md_heading3",
    "md_heading4",
    "md_bold",
    "md_italic",
    "md_link",
    "md_link_url",
    "md_code_inline",
    "md_code_block_bg",
    "md_code_block_fg",
    "md_blockquote",
    "md_list_marker",
    "md_hr",
]

SYNTAX_KEYS: list[str] = [
    "syntax_keyword",
    "syntax_string",
    "syntax_number",
    "syntax_comment",
    "syntax_function",
    "syntax_variable",
    "syntax_operator",
    "syntax_type",
    "syntax_constant",
    "syntax_punctuation",
]

INPUT_KEYS: list[str] = [
    "input_bg",
    "input_fg",
    "input_placeholder",
    "input_cursor",
    "input_selection",
]

COMPONENT_KEYS: list[str] = [
    "scrollbar_track",
    "scrollbar_thumb",
    "menu_bg",
    "menu_fg",
    "menu_selected_bg",
    "menu_selected_fg",
    "overlay_bg",
    "overlay_border",
    "status_bar_bg",
    "status_bar_fg",
]

ALL_COLOR_KEYS: list[str] = (
    CORE_UI_KEYS + MESSAGE_KEYS + MARKDOWN_KEYS + SYNTAX_KEYS + INPUT_KEYS + COMPONENT_KEYS
)
"""Complete list of recognised colour keys (~53 keys)."""


# ---------------------------------------------------------------------------
# ThemeColor convenience alias
# ---------------------------------------------------------------------------

ThemeColor = dict[str, str]
"""A mapping from colour key names to hex colour strings."""


# ---------------------------------------------------------------------------
# ThemeInfo
# ---------------------------------------------------------------------------


@dataclass
class ThemeInfo:
    """
    Full theme definition including metadata and resolved colours.

    Attributes
    ----------
    name:
        Short identifier for the theme (e.g. ``"default-dark"``).
    description:
        One-line human-readable description.
    author:
        Theme author name or handle.
    colors:
        Mapping from colour key to hex colour string.  All keys listed in
        :data:`ALL_COLOR_KEYS` should be present after loading; the loader
        fills in defaults for any missing keys.
    """

    name: str = "untitled"
    description: str = ""
    author: str = ""
    colors: ThemeColor = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, key: str, fallback: str = "#ffffff") -> str:
        """Return the colour for *key*, or *fallback* if absent."""
        return self.colors.get(key, fallback)

    def __getitem__(self, key: str) -> str:
        return self.colors[key]

    def __contains__(self, key: str) -> bool:
        return key in self.colors
