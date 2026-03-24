"""Tests for autocomplete system."""

from unittest.mock import patch

import pytest

from skillengine.tui.autocomplete import (
    CombinedAutocomplete,
    CommandAutocomplete,
    FileAutocomplete,
    SlashCommand,
    Suggestion,
)


class TestSuggestion:
    """Tests for Suggestion dataclass."""

    def test_creation(self) -> None:
        """Should create a suggestion with all fields."""
        s = Suggestion(text="hello", display="Hello!", description="A greeting")

        assert s.text == "hello"
        assert s.display == "Hello!"
        assert s.description == "A greeting"

    def test_display_defaults_to_text(self) -> None:
        """When display is not provided, it should default to text."""
        s = Suggestion(text="world")

        assert s.display == "world"

    def test_frozen(self) -> None:
        """Suggestion should be immutable (frozen dataclass)."""
        s = Suggestion(text="test")

        with pytest.raises(AttributeError):
            s.text = "other"  # type: ignore[misc]


class TestCommandAutocomplete:
    """Tests for CommandAutocomplete."""

    def test_no_prefix_returns_empty(self) -> None:
        """Input without '/' prefix should return no suggestions."""
        commands = [SlashCommand(name="help", description="Show help")]
        provider = CommandAutocomplete(commands=commands)

        assert provider.get_suggestions("help") == []
        assert provider.get_suggestions("") == []

    def test_slash_prefix_returns_matching_commands(self) -> None:
        """'/' prefix should return matching slash commands."""
        commands = [
            SlashCommand(name="help", description="Show help"),
            SlashCommand(name="quit", description="Quit"),
        ]
        provider = CommandAutocomplete(commands=commands)
        results = provider.get_suggestions("/")

        assert len(results) == 2
        assert results[0].text == "/help"
        assert results[1].text == "/quit"

    def test_filters_by_prefix(self) -> None:
        """Should only return commands whose names start with the query."""
        commands = [
            SlashCommand(name="help", description="Show help"),
            SlashCommand(name="history", description="Show history"),
            SlashCommand(name="quit", description="Quit"),
        ]
        provider = CommandAutocomplete(commands=commands)
        results = provider.get_suggestions("/h")

        assert len(results) == 2
        texts = [r.text for r in results]
        assert "/help" in texts
        assert "/history" in texts
        assert "/quit" not in texts

    def test_no_matching_commands(self) -> None:
        """Should return empty list when no commands match."""
        commands = [SlashCommand(name="help")]
        provider = CommandAutocomplete(commands=commands)

        assert provider.get_suggestions("/zzz") == []

    def test_empty_commands_list(self) -> None:
        """Provider with no commands should return empty for any input."""
        provider = CommandAutocomplete()

        assert provider.get_suggestions("/anything") == []


class TestFileAutocomplete:
    """Tests for FileAutocomplete."""

    def test_non_at_prefix_returns_empty(self) -> None:
        """Input without '@' prefix should return no suggestions."""
        provider = FileAutocomplete()

        assert provider.get_suggestions("somefile") == []
        assert provider.get_suggestions("") == []

    def test_at_prefix_returns_files(self, tmp_path) -> None:
        """'@' prefix should return file suggestions from cwd."""
        # Create test files
        (tmp_path / "readme.md").write_text("readme")
        (tmp_path / "setup.py").write_text("setup")
        (tmp_path / "config.yaml").write_text("config")

        # Patch _git_ls_files to return None so it falls back to os.walk
        provider = FileAutocomplete(cwd=tmp_path, max_results=50)
        with patch.object(provider, "_git_ls_files", return_value=None):
            results = provider.get_suggestions("@")

        assert len(results) == 3
        texts = [r.text for r in results]
        assert "@config.yaml" in texts
        assert "@readme.md" in texts
        assert "@setup.py" in texts

    def test_at_prefix_filters_by_query(self, tmp_path) -> None:
        """'@' prefix with a query should filter matching files."""
        (tmp_path / "alpha.py").write_text("")
        (tmp_path / "beta.py").write_text("")
        (tmp_path / "gamma.txt").write_text("")

        provider = FileAutocomplete(cwd=tmp_path, max_results=50)
        with patch.object(provider, "_git_ls_files", return_value=None):
            results = provider.get_suggestions("@alpha")

        assert len(results) == 1
        assert results[0].text == "@alpha.py"

    def test_max_results_limit(self, tmp_path) -> None:
        """Should respect the max_results limit."""
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")

        provider = FileAutocomplete(cwd=tmp_path, max_results=3)
        with patch.object(provider, "_git_ls_files", return_value=None):
            results = provider.get_suggestions("@")

        assert len(results) == 3


class TestCombinedAutocomplete:
    """Tests for CombinedAutocomplete."""

    def test_non_merge_returns_first_provider_results(self) -> None:
        """In non-merge mode, should return results from the first provider that matches."""
        cmd_provider = CommandAutocomplete(
            commands=[SlashCommand(name="help", description="Help")]
        )
        cmd_provider_2 = CommandAutocomplete(
            commands=[SlashCommand(name="quit", description="Quit")]
        )
        combined = CombinedAutocomplete(
            providers=[cmd_provider, cmd_provider_2], merge=False
        )

        results = combined.get_suggestions("/")

        # First provider has a match, so second provider is not consulted
        assert len(results) == 1
        assert results[0].text == "/help"

    def test_non_merge_falls_through_to_next_provider(self) -> None:
        """Non-merge mode should try the next provider if the first returns nothing."""
        cmd_provider_empty = CommandAutocomplete(commands=[])
        cmd_provider = CommandAutocomplete(
            commands=[SlashCommand(name="help", description="Help")]
        )
        combined = CombinedAutocomplete(
            providers=[cmd_provider_empty, cmd_provider], merge=False
        )

        results = combined.get_suggestions("/")

        assert len(results) == 1
        assert results[0].text == "/help"

    def test_merge_mode_combines_results(self) -> None:
        """In merge mode, results from all providers should be combined."""
        cmd_provider_1 = CommandAutocomplete(
            commands=[SlashCommand(name="help")]
        )
        cmd_provider_2 = CommandAutocomplete(
            commands=[SlashCommand(name="quit")]
        )
        combined = CombinedAutocomplete(
            providers=[cmd_provider_1, cmd_provider_2], merge=True
        )

        results = combined.get_suggestions("/")

        assert len(results) == 2
        texts = [r.text for r in results]
        assert "/help" in texts
        assert "/quit" in texts

    def test_merge_mode_respects_max_results(self) -> None:
        """Merge mode should cap at max_results."""
        commands = [SlashCommand(name=f"cmd{i}") for i in range(10)]
        provider = CommandAutocomplete(commands=commands)
        combined = CombinedAutocomplete(
            providers=[provider], merge=True, max_results=3
        )

        results = combined.get_suggestions("/")

        assert len(results) == 3

    def test_empty_prefix_returns_empty(self) -> None:
        """Empty prefix should return empty when no provider handles it."""
        cmd_provider = CommandAutocomplete(
            commands=[SlashCommand(name="help")]
        )
        combined = CombinedAutocomplete(providers=[cmd_provider])

        results = combined.get_suggestions("")

        assert results == []

    def test_no_providers_returns_empty(self) -> None:
        """With no providers, should always return empty."""
        combined = CombinedAutocomplete()

        assert combined.get_suggestions("/help") == []
        assert combined.get_suggestions("@file") == []
