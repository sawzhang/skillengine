"""
Built-in default dark theme.

Provides a complete set of sensible colours for every key defined in
:data:`~skillengine.tui.theme.models.ALL_COLOR_KEYS`.
"""

from __future__ import annotations

from skillengine.tui.theme.models import ThemeColor, ThemeInfo

# ---------------------------------------------------------------------------
# Default dark palette
# ---------------------------------------------------------------------------

DEFAULT_DARK_COLORS: ThemeColor = {
    # Core UI
    "primary": "#7aa2f7",
    "secondary": "#9ece6a",
    "accent": "#bb9af7",
    "background": "#1a1b26",
    "surface": "#24283b",
    "surface_variant": "#292e42",
    "border": "#3b4261",
    "border_focused": "#7aa2f7",
    "text": "#c0caf5",
    "text_muted": "#a9b1d6",
    "text_dim": "#565f89",
    "text_inverse": "#1a1b26",
    "error": "#f7768e",
    "warning": "#e0af68",
    "success": "#9ece6a",
    "info": "#7dcfff",
    # Message backgrounds
    "message_user_bg": "#1a1b26",
    "message_user_fg": "#c0caf5",
    "message_assistant_bg": "#24283b",
    "message_assistant_fg": "#c0caf5",
    "message_system_bg": "#292e42",
    "message_system_fg": "#a9b1d6",
    "message_tool_bg": "#1a1b26",
    "message_tool_fg": "#7dcfff",
    # Markdown rendering
    "md_heading1": "#ff9e64",
    "md_heading2": "#7aa2f7",
    "md_heading3": "#7dcfff",
    "md_heading4": "#bb9af7",
    "md_bold": "#c0caf5",
    "md_italic": "#a9b1d6",
    "md_link": "#7aa2f7",
    "md_link_url": "#565f89",
    "md_code_inline": "#9ece6a",
    "md_code_block_bg": "#292e42",
    "md_code_block_fg": "#c0caf5",
    "md_blockquote": "#565f89",
    "md_list_marker": "#e0af68",
    "md_hr": "#3b4261",
    # Syntax highlighting
    "syntax_keyword": "#bb9af7",
    "syntax_string": "#9ece6a",
    "syntax_number": "#ff9e64",
    "syntax_comment": "#565f89",
    "syntax_function": "#7aa2f7",
    "syntax_variable": "#c0caf5",
    "syntax_operator": "#89ddff",
    "syntax_type": "#2ac3de",
    "syntax_constant": "#ff9e64",
    "syntax_punctuation": "#a9b1d6",
    # Input widgets
    "input_bg": "#292e42",
    "input_fg": "#c0caf5",
    "input_placeholder": "#565f89",
    "input_cursor": "#c0caf5",
    "input_selection": "#3b4261",
    # Component chrome
    "scrollbar_track": "#1a1b26",
    "scrollbar_thumb": "#3b4261",
    "menu_bg": "#24283b",
    "menu_fg": "#c0caf5",
    "menu_selected_bg": "#3b4261",
    "menu_selected_fg": "#7aa2f7",
    "overlay_bg": "#24283b",
    "overlay_border": "#3b4261",
    "status_bar_bg": "#1a1b26",
    "status_bar_fg": "#a9b1d6",
}


DEFAULT_DARK_THEME = ThemeInfo(
    name="default-dark",
    description="Built-in dark theme inspired by Tokyo Night",
    author="skillengine",
    colors=dict(DEFAULT_DARK_COLORS),
)
"""Pre-built default dark theme instance."""


def get_default_theme() -> ThemeInfo:
    """
    Return a fresh copy of the default dark theme.

    A copy is returned so that callers can mutate it without affecting
    the module-level constant.
    """
    return ThemeInfo(
        name=DEFAULT_DARK_THEME.name,
        description=DEFAULT_DARK_THEME.description,
        author=DEFAULT_DARK_THEME.author,
        colors=dict(DEFAULT_DARK_THEME.colors),
    )
