"""
Autocomplete system.

Provides pluggable autocomplete providers for file paths (``@`` prefix),
slash commands (``/`` prefix), and a combiner that merges results from
multiple providers.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Suggestion model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suggestion:
    """
    A single autocomplete suggestion.

    Attributes
    ----------
    text:
        The value to insert into the input.
    display:
        The text shown in the completion menu (may include formatting).
    description:
        Optional short description shown alongside the item.
    """

    text: str
    display: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        # Default *display* to *text* when not explicitly set.
        if not self.display:
            object.__setattr__(self, "display", self.text)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class AutocompleteProvider(ABC):
    """
    Base class for autocomplete providers.

    Subclasses must implement :meth:`get_suggestions` which returns an
    ordered list of suggestions for a given input prefix.
    """

    @abstractmethod
    def get_suggestions(self, prefix: str) -> list[Suggestion]:
        """
        Return suggestions for *prefix*.

        Parameters
        ----------
        prefix:
            The current (partial) text in the input widget.

        Returns
        -------
        list[Suggestion]
            Ordered list of completions.
        """
        ...


# ---------------------------------------------------------------------------
# File autocomplete  (triggered by @<path>)
# ---------------------------------------------------------------------------


class FileAutocomplete(AutocompleteProvider):
    """
    Suggest file paths when the user types an ``@`` prefix.

    Uses ``git ls-files`` when inside a Git repository so that files
    matched by ``.gitignore`` are excluded.  Falls back to :func:`os.walk`
    outside of a repository.

    Parameters
    ----------
    cwd:
        Working directory to resolve paths relative to.  Defaults to the
        current working directory at construction time.
    max_results:
        Maximum number of suggestions to return.
    """

    def __init__(self, cwd: str | Path | None = None, max_results: int = 50) -> None:
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._max_results = max_results

    def get_suggestions(self, prefix: str) -> list[Suggestion]:
        """Return file path suggestions for an ``@``-prefixed query."""
        if not prefix.startswith("@"):
            return []

        query = prefix[1:]  # strip the leading '@'
        files = self._list_files(query)
        return files[: self._max_results]

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    def _list_files(self, query: str) -> list[Suggestion]:
        """
        List files matching *query*.

        Tries ``git ls-files`` first for .gitignore awareness, then falls
        back to ``os.walk``.
        """
        files = self._git_ls_files(query)
        if files is not None:
            return files
        return self._walk_files(query)

    def _git_ls_files(self, query: str) -> list[Suggestion] | None:
        """
        Use ``git ls-files`` to list tracked files matching *query*.

        Returns ``None`` if not inside a Git repo or the command fails.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                cwd=str(self._cwd),
                timeout=5,
            )
            if result.returncode != 0:
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

        query_lower = query.lower()
        suggestions: list[Suggestion] = []

        for line in result.stdout.splitlines():
            path_str = line.strip()
            if not path_str:
                continue
            if query_lower and query_lower not in path_str.lower():
                continue

            name = Path(path_str).name
            suggestions.append(
                Suggestion(
                    text=f"@{path_str}",
                    display=path_str,
                    description=name,
                )
            )

        # Sort by relevance: paths starting with the query first, then alphabetical
        suggestions.sort(
            key=lambda s: (
                not s.display.lower().startswith(query_lower),
                s.display.lower(),
            )
        )
        return suggestions

    def _walk_files(self, query: str) -> list[Suggestion]:
        """Fall-back file listing using :func:`os.walk`."""
        query_lower = query.lower()
        suggestions: list[Suggestion] = []
        base = self._cwd

        for dirpath, dirnames, filenames in os.walk(base):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            rel_dir = Path(dirpath).relative_to(base)
            for fname in filenames:
                if fname.startswith("."):
                    continue
                rel_path = str(rel_dir / fname) if str(rel_dir) != "." else fname
                if query_lower and query_lower not in rel_path.lower():
                    continue
                suggestions.append(
                    Suggestion(
                        text=f"@{rel_path}",
                        display=rel_path,
                        description=fname,
                    )
                )
                if len(suggestions) >= self._max_results * 2:
                    break
            if len(suggestions) >= self._max_results * 2:
                break

        suggestions.sort(key=lambda s: s.display.lower())
        return suggestions[: self._max_results]


# ---------------------------------------------------------------------------
# Command autocomplete  (triggered by /<command>)
# ---------------------------------------------------------------------------


@dataclass
class SlashCommand:
    """Definition of a slash command for autocomplete."""

    name: str
    description: str = ""


class CommandAutocomplete(AutocompleteProvider):
    """
    Suggest slash commands when the user types a ``/`` prefix.

    Parameters
    ----------
    commands:
        List of available slash command definitions.
    """

    def __init__(self, commands: list[SlashCommand] | None = None) -> None:
        self._commands: list[SlashCommand] = list(commands) if commands else []

    @property
    def commands(self) -> list[SlashCommand]:
        return self._commands

    @commands.setter
    def commands(self, value: list[SlashCommand]) -> None:
        self._commands = list(value)

    def get_suggestions(self, prefix: str) -> list[Suggestion]:
        """Return matching slash commands."""
        if not prefix.startswith("/"):
            return []

        query = prefix[1:].lower()
        results: list[Suggestion] = []

        for cmd in self._commands:
            if query and not cmd.name.lower().startswith(query):
                continue
            results.append(
                Suggestion(
                    text=f"/{cmd.name}",
                    display=f"/{cmd.name}",
                    description=cmd.description,
                )
            )

        results.sort(key=lambda s: s.text.lower())
        return results


# ---------------------------------------------------------------------------
# Combined autocomplete
# ---------------------------------------------------------------------------


class CombinedAutocomplete(AutocompleteProvider):
    """
    Merges suggestions from multiple providers.

    The providers are tried in order; the first provider that returns a
    non-empty list of suggestions wins (i.e. results are *not* merged
    across providers).  If you want merged results, set
    ``merge=True`` at construction time.

    Parameters
    ----------
    providers:
        Ordered list of providers to consult.
    merge:
        If ``True``, combine results from all providers instead of
        stopping at the first match.
    max_results:
        Maximum total suggestions returned.
    """

    def __init__(
        self,
        providers: list[AutocompleteProvider] | None = None,
        merge: bool = False,
        max_results: int = 50,
    ) -> None:
        self._providers: list[AutocompleteProvider] = list(providers) if providers else []
        self._merge = merge
        self._max_results = max_results

    @property
    def providers(self) -> list[AutocompleteProvider]:
        return self._providers

    def add_provider(self, provider: AutocompleteProvider) -> None:
        """Append a provider to the chain."""
        self._providers.append(provider)

    def get_suggestions(self, prefix: str) -> list[Suggestion]:
        """
        Return suggestions by delegating to child providers.

        In non-merge mode the first provider that returns results wins.
        In merge mode results are concatenated.
        """
        if self._merge:
            all_suggestions: list[Suggestion] = []
            for provider in self._providers:
                all_suggestions.extend(provider.get_suggestions(prefix))
            return all_suggestions[: self._max_results]

        for provider in self._providers:
            results = provider.get_suggestions(prefix)
            if results:
                return results[: self._max_results]

        return []
