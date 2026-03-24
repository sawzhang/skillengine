"""
Markdown rendering component.

Uses the ``rich`` library to render Markdown content with syntax highlighting,
headers, lists, bold, italic, links, and fenced code blocks.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.theme import Theme as RichTheme

from skillengine.tui.component import Component
from skillengine.tui.keys import Key

# ---------------------------------------------------------------------------
# Default code-block theme overrides (keep it minimal)
# ---------------------------------------------------------------------------

_RICH_THEME = RichTheme(
    {
        "markdown.h1": "bold bright_white",
        "markdown.h2": "bold bright_cyan",
        "markdown.h3": "bold cyan",
        "markdown.h4": "bold dim cyan",
        "markdown.link": "bright_blue underline",
        "markdown.link_url": "dim blue",
        "markdown.code": "bright_green on grey11",
        "markdown.item.bullet": "bright_yellow",
    }
)


class MarkdownWidget(Component):
    """
    Read-only component that renders Markdown text using ``rich``.

    Example
    -------
    >>> widget = MarkdownWidget()
    >>> widget.set_content("# Hello\\n\\nSome **bold** text.")
    """

    def __init__(self) -> None:
        super().__init__()
        self._content: str = ""
        self._rendered_lines: list[str] = []
        self._render_width: int = 0  # width used for the last render cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_content(self, markdown_text: str) -> None:
        """Replace the displayed Markdown content."""
        if markdown_text != self._content:
            self._content = markdown_text
            self._rendered_lines = []  # invalidate cache
            self._render_width = 0
            self.invalidate()

    @property
    def content(self) -> str:
        """The raw Markdown source text."""
        return self._content

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """
        Render Markdown to styled terminal lines.

        Results are cached and only re-rendered when the content or width
        changes.
        """
        effective_width = max(10, width)

        if self._rendered_lines and self._render_width == effective_width and not self._dirty:
            return self._rendered_lines

        self._rendered_lines = self._render_markdown(self._content, effective_width)
        self._render_width = effective_width
        self._dirty = False
        return self._rendered_lines

    def handle_input(self, key: Key) -> bool:
        """Markdown widget is read-only; no input handling."""
        return False

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_markdown(text: str, width: int) -> list[str]:
        """
        Use ``rich`` to render Markdown into a list of terminal lines.

        The ``Console`` is configured with ``force_terminal=True`` so that
        ANSI sequences are produced even when stdout is not a TTY (e.g.
        when the output is captured into a ``StringIO``).
        """
        if not text.strip():
            return [""]

        buf = StringIO()
        console = Console(
            file=buf,
            width=width,
            force_terminal=True,
            no_color=False,
            highlight=False,
            theme=_RICH_THEME,
        )

        md = RichMarkdown(text, code_theme="monokai")
        console.print(md)

        raw = buf.getvalue()
        # Split into lines, stripping the final trailing newline that rich adds
        lines = raw.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        return lines if lines else [""]
