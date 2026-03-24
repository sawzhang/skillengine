"""Tests for keybinding management."""

import json

import pytest

from skillengine.tui.keybindings import DEFAULT_KEYBINDINGS, KeybindingsManager
from skillengine.tui.keys import Key


class TestKeybindingsManager:
    """Tests for KeybindingsManager."""

    def test_defaults_loaded(self) -> None:
        """All default actions should be present in a fresh manager."""
        manager = KeybindingsManager()
        actions = manager.actions()

        for action in DEFAULT_KEYBINDINGS:
            assert action in actions

    def test_matches_with_string_descriptor(self) -> None:
        """String descriptor 'ctrl+c' should match the 'interrupt' action."""
        manager = KeybindingsManager()

        assert manager.matches("ctrl+c", "interrupt") is True
        assert manager.matches("ctrl+d", "interrupt") is False

    def test_matches_with_key_object(self) -> None:
        """A Key object with ctrl=True and name='ctrl+c' should match 'interrupt'."""
        manager = KeybindingsManager()
        key = Key(name="ctrl+c", char="c", ctrl=True)

        assert manager.matches(key, "interrupt") is True

    def test_matches_shift_tab_with_key_object(self) -> None:
        """A Key with shift=True and name='tab' should match 'cycle_thinking'."""
        manager = KeybindingsManager()
        key = Key(name="tab", char="\t", shift=True)

        assert manager.matches(key, "cycle_thinking") is True

    def test_user_overrides_replace_defaults(self) -> None:
        """User overrides should replace the default bindings for that action."""
        manager = KeybindingsManager(user_overrides={"interrupt": ["ctrl+x"]})

        assert manager.matches("ctrl+x", "interrupt") is True
        assert manager.matches("ctrl+c", "interrupt") is False

    def test_user_overrides_preserve_other_defaults(self) -> None:
        """Overriding one action should not affect other defaults."""
        manager = KeybindingsManager(user_overrides={"interrupt": ["ctrl+x"]})

        assert manager.matches("ctrl+d", "exit") is True

    def test_get_keys_returns_descriptors(self) -> None:
        """get_keys should return the original descriptor strings for an action."""
        manager = KeybindingsManager()
        keys = manager.get_keys("interrupt")

        assert keys == ["ctrl+c"]

    def test_get_keys_unknown_action(self) -> None:
        """get_keys for an unknown action should return an empty list."""
        manager = KeybindingsManager()

        assert manager.get_keys("nonexistent") == []

    def test_actions_returns_all_action_names(self) -> None:
        """actions() should return all registered action names."""
        manager = KeybindingsManager()
        actions = manager.actions()

        assert set(actions) == set(DEFAULT_KEYBINDINGS.keys())

    def test_find_action_returns_first_match(self) -> None:
        """find_action should return the action name for a matching key."""
        manager = KeybindingsManager()

        assert manager.find_action("ctrl+c") == "interrupt"
        assert manager.find_action("ctrl+d") == "exit"

    def test_find_action_returns_none_for_unknown_key(self) -> None:
        """find_action should return None when no action matches."""
        manager = KeybindingsManager()

        assert manager.find_action("ctrl+z") is None

    def test_load_from_json_file(self, tmp_path) -> None:
        """load() should read overrides from a JSON config file."""
        config_file = tmp_path / "keybindings.json"
        config_data = {
            "exit": ["ctrl+q", "ctrl+d"],
            "interrupt": ["ctrl+z"],
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        manager = KeybindingsManager.load(config_path=config_file)

        assert manager.matches("ctrl+q", "exit") is True
        assert manager.matches("ctrl+d", "exit") is True
        assert manager.matches("ctrl+z", "interrupt") is True
        # Original ctrl+c should be replaced
        assert manager.matches("ctrl+c", "interrupt") is False

    def test_load_with_missing_file_uses_defaults(self, tmp_path) -> None:
        """load() with a nonexistent path should fall back to defaults."""
        missing = tmp_path / "does_not_exist.json"

        manager = KeybindingsManager.load(config_path=missing)

        assert manager.matches("ctrl+c", "interrupt") is True
        assert set(manager.actions()) == set(DEFAULT_KEYBINDINGS.keys())
