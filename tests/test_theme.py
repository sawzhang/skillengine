"""Tests for the TUI theme system: models, defaults, loader, schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillengine.tui.theme import (
    ALL_COLOR_KEYS,
    DEFAULT_DARK_THEME,
    THEME_SCHEMA,
    ThemeInfo,
    discover_themes,
    get_default_theme,
    load_theme,
    validate_theme,
)


# ---------------------------------------------------------------------------
# TestThemeInfo
# ---------------------------------------------------------------------------


class TestThemeInfo:
    """Tests for the ThemeInfo dataclass."""

    def test_creation_with_defaults(self) -> None:
        theme = ThemeInfo()
        assert theme.name == "untitled"
        assert theme.description == ""
        assert theme.author == ""
        assert theme.colors == {}

    def test_creation_with_values(self) -> None:
        colors = {"primary": "#ff0000", "background": "#000000"}
        theme = ThemeInfo(
            name="test-theme",
            description="A test theme",
            author="tester",
            colors=colors,
        )
        assert theme.name == "test-theme"
        assert theme.description == "A test theme"
        assert theme.author == "tester"
        assert theme.colors == colors

    def test_get_existing_key(self) -> None:
        theme = ThemeInfo(colors={"primary": "#abcdef"})
        assert theme.get("primary") == "#abcdef"

    def test_get_missing_key_returns_fallback(self) -> None:
        theme = ThemeInfo(colors={})
        assert theme.get("missing") == "#ffffff"

    def test_get_custom_fallback(self) -> None:
        theme = ThemeInfo(colors={})
        assert theme.get("missing", fallback="#000000") == "#000000"

    def test_getitem_existing(self) -> None:
        theme = ThemeInfo(colors={"primary": "#112233"})
        assert theme["primary"] == "#112233"

    def test_getitem_missing_raises(self) -> None:
        theme = ThemeInfo(colors={})
        with pytest.raises(KeyError):
            _ = theme["nonexistent"]

    def test_contains_existing_key(self) -> None:
        theme = ThemeInfo(colors={"primary": "#abc"})
        assert "primary" in theme

    def test_contains_missing_key(self) -> None:
        theme = ThemeInfo(colors={})
        assert "primary" not in theme


# ---------------------------------------------------------------------------
# TestAllColorKeys
# ---------------------------------------------------------------------------


class TestAllColorKeys:
    """Tests for the ALL_COLOR_KEYS list."""

    def test_not_empty(self) -> None:
        assert len(ALL_COLOR_KEYS) > 0

    def test_contains_primary(self) -> None:
        assert "primary" in ALL_COLOR_KEYS

    def test_contains_background(self) -> None:
        assert "background" in ALL_COLOR_KEYS

    def test_contains_secondary(self) -> None:
        assert "secondary" in ALL_COLOR_KEYS

    def test_contains_error(self) -> None:
        assert "error" in ALL_COLOR_KEYS

    def test_contains_text(self) -> None:
        assert "text" in ALL_COLOR_KEYS

    def test_contains_syntax_keys(self) -> None:
        assert "syntax_keyword" in ALL_COLOR_KEYS
        assert "syntax_string" in ALL_COLOR_KEYS

    def test_contains_input_keys(self) -> None:
        assert "input_bg" in ALL_COLOR_KEYS
        assert "input_fg" in ALL_COLOR_KEYS

    def test_contains_message_keys(self) -> None:
        assert "message_user_bg" in ALL_COLOR_KEYS
        assert "message_assistant_bg" in ALL_COLOR_KEYS

    def test_approximate_count(self) -> None:
        # The spec says ~53 keys
        assert len(ALL_COLOR_KEYS) >= 50

    def test_all_strings(self) -> None:
        for key in ALL_COLOR_KEYS:
            assert isinstance(key, str)

    def test_no_duplicates(self) -> None:
        assert len(ALL_COLOR_KEYS) == len(set(ALL_COLOR_KEYS))


# ---------------------------------------------------------------------------
# TestDefaultTheme
# ---------------------------------------------------------------------------


class TestDefaultTheme:
    """Tests for DEFAULT_DARK_THEME and get_default_theme."""

    def test_default_theme_name(self) -> None:
        assert DEFAULT_DARK_THEME.name == "default-dark"

    def test_default_theme_has_description(self) -> None:
        assert DEFAULT_DARK_THEME.description != ""

    def test_default_theme_has_author(self) -> None:
        assert DEFAULT_DARK_THEME.author != ""

    def test_default_theme_has_all_color_keys(self) -> None:
        missing = [
            key for key in ALL_COLOR_KEYS if key not in DEFAULT_DARK_THEME.colors
        ]
        assert missing == [], f"Missing color keys: {missing}"

    def test_default_theme_colors_are_hex(self) -> None:
        for key, value in DEFAULT_DARK_THEME.colors.items():
            assert isinstance(value, str), f"Color for '{key}' is not a string"
            assert value.startswith("#"), f"Color for '{key}' does not start with #: {value}"

    def test_get_default_theme_returns_copy(self) -> None:
        copy = get_default_theme()
        assert copy.name == DEFAULT_DARK_THEME.name
        assert copy.colors == DEFAULT_DARK_THEME.colors
        # Modifying the copy should not affect the original
        copy.colors["primary"] = "#000000"
        assert DEFAULT_DARK_THEME.colors["primary"] != "#000000"

    def test_get_default_theme_returns_new_instance(self) -> None:
        copy1 = get_default_theme()
        copy2 = get_default_theme()
        assert copy1 is not copy2
        assert copy1.colors is not copy2.colors

    def test_get_default_theme_preserves_metadata(self) -> None:
        copy = get_default_theme()
        assert copy.description == DEFAULT_DARK_THEME.description
        assert copy.author == DEFAULT_DARK_THEME.author


# ---------------------------------------------------------------------------
# TestLoadTheme
# ---------------------------------------------------------------------------


class TestLoadTheme:
    """Tests for loading themes from JSON files."""

    def test_load_basic_theme(self, tmp_path: Path) -> None:
        theme_data = {
            "name": "my-theme",
            "description": "My custom theme",
            "author": "test-author",
            "colors": {
                "primary": "#ff0000",
                "background": "#000000",
            },
        }
        theme_file = tmp_path / "my-theme.json"
        theme_file.write_text(json.dumps(theme_data))

        theme = load_theme(theme_file)
        assert theme.name == "my-theme"
        assert theme.description == "My custom theme"
        assert theme.author == "test-author"
        assert theme.colors["primary"] == "#ff0000"
        assert theme.colors["background"] == "#000000"

    def test_load_theme_name_defaults_to_stem(self, tmp_path: Path) -> None:
        theme_data = {"colors": {"primary": "#aabbcc"}}
        theme_file = tmp_path / "fallback-name.json"
        theme_file.write_text(json.dumps(theme_data))

        theme = load_theme(theme_file)
        assert theme.name == "fallback-name"

    def test_load_theme_with_variables(self, tmp_path: Path) -> None:
        theme_data = {
            "name": "var-theme",
            "variables": {
                "blue": "#0000ff",
                "red": "#ff0000",
            },
            "colors": {
                "primary": "blue",
                "error": "red",
                "background": "#111111",
            },
        }
        theme_file = tmp_path / "var-theme.json"
        theme_file.write_text(json.dumps(theme_data))

        theme = load_theme(theme_file)
        assert theme.colors["primary"] == "#0000ff"
        assert theme.colors["error"] == "#ff0000"
        assert theme.colors["background"] == "#111111"

    def test_load_theme_missing_optional_fields(self, tmp_path: Path) -> None:
        theme_data = {
            "name": "minimal",
            "colors": {},
        }
        theme_file = tmp_path / "minimal.json"
        theme_file.write_text(json.dumps(theme_data))

        theme = load_theme(theme_file)
        assert theme.name == "minimal"
        assert theme.description == ""
        assert theme.author == ""
        assert theme.colors == {}

    def test_load_theme_returns_theme_info(self, tmp_path: Path) -> None:
        theme_data = {"name": "type-check", "colors": {"primary": "#aaa"}}
        theme_file = tmp_path / "type-check.json"
        theme_file.write_text(json.dumps(theme_data))

        theme = load_theme(theme_file)
        assert isinstance(theme, ThemeInfo)


# ---------------------------------------------------------------------------
# TestDiscoverThemes
# ---------------------------------------------------------------------------


class TestDiscoverThemes:
    """Tests for discovering theme files from directories."""

    def test_discover_from_user_dir(self, tmp_path: Path) -> None:
        themes_dir = tmp_path / "user-themes"
        themes_dir.mkdir()
        (themes_dir / "dark.json").write_text('{"name": "dark", "colors": {}}')
        (themes_dir / "light.json").write_text('{"name": "light", "colors": {}}')

        found = discover_themes(user_dir=themes_dir, project_dir=tmp_path / "empty")
        assert len(found) == 2
        names = {p.stem for p in found}
        assert "dark" in names
        assert "light" in names

    def test_discover_from_project_dir(self, tmp_path: Path) -> None:
        themes_dir = tmp_path / "project-themes"
        themes_dir.mkdir()
        (themes_dir / "custom.json").write_text('{"name": "custom", "colors": {}}')

        found = discover_themes(
            user_dir=tmp_path / "nonexistent",
            project_dir=themes_dir,
        )
        assert len(found) == 1
        assert found[0].stem == "custom"

    def test_discover_from_both_dirs(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (user_dir / "user-theme.json").write_text('{"name": "u", "colors": {}}')

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "project-theme.json").write_text('{"name": "p", "colors": {}}')

        found = discover_themes(user_dir=user_dir, project_dir=project_dir)
        assert len(found) == 2

    def test_discover_ignores_non_json_files(self, tmp_path: Path) -> None:
        themes_dir = tmp_path / "mixed"
        themes_dir.mkdir()
        (themes_dir / "valid.json").write_text('{"name": "valid", "colors": {}}')
        (themes_dir / "readme.txt").write_text("not a theme")
        (themes_dir / "config.yaml").write_text("also not a theme")

        found = discover_themes(user_dir=themes_dir, project_dir=tmp_path / "empty")
        assert len(found) == 1
        assert found[0].name == "valid.json"

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        themes_dir = tmp_path / "empty"
        themes_dir.mkdir()

        found = discover_themes(user_dir=themes_dir, project_dir=tmp_path / "also-empty")
        assert found == []

    def test_discover_nonexistent_directories(self, tmp_path: Path) -> None:
        found = discover_themes(
            user_dir=tmp_path / "no-such-dir",
            project_dir=tmp_path / "also-no-such-dir",
        )
        assert found == []

    def test_discover_returns_path_objects(self, tmp_path: Path) -> None:
        themes_dir = tmp_path / "paths"
        themes_dir.mkdir()
        (themes_dir / "theme.json").write_text('{"name": "t", "colors": {}}')

        found = discover_themes(user_dir=themes_dir, project_dir=tmp_path / "empty")
        assert all(isinstance(p, Path) for p in found)


# ---------------------------------------------------------------------------
# TestValidateTheme
# ---------------------------------------------------------------------------


class TestValidateTheme:
    """Tests for theme validation."""

    def test_valid_theme_no_errors(self) -> None:
        data = {
            "name": "valid-theme",
            "colors": {
                "primary": "#ff0000",
                "background": "#000000",
            },
        }
        errors = validate_theme(data)
        assert errors == []

    def test_missing_name_returns_error(self) -> None:
        data = {"colors": {"primary": "#ff0000"}}
        errors = validate_theme(data)
        assert any("name" in e.lower() for e in errors)

    def test_missing_colors_returns_error(self) -> None:
        data = {"name": "no-colors"}
        errors = validate_theme(data)
        assert any("colors" in e.lower() for e in errors)

    def test_missing_both_required_fields(self) -> None:
        data: dict = {}
        errors = validate_theme(data)
        assert len(errors) >= 2
        error_text = " ".join(errors).lower()
        assert "name" in error_text
        assert "colors" in error_text

    def test_invalid_hex_color_returns_error(self) -> None:
        data = {
            "name": "bad-hex",
            "colors": {
                "primary": "#xyz123",
            },
        }
        errors = validate_theme(data)
        assert any("hex" in e.lower() or "invalid" in e.lower() for e in errors)

    def test_invalid_hex_color_wrong_length(self) -> None:
        data = {
            "name": "bad-length",
            "colors": {
                "primary": "#1234",  # 4 hex chars is invalid (must be 3, 6, or 8)
            },
        }
        errors = validate_theme(data)
        assert any("hex" in e.lower() or "invalid" in e.lower() for e in errors)

    def test_valid_hex_3_digits(self) -> None:
        data = {"name": "short-hex", "colors": {"primary": "#abc"}}
        errors = validate_theme(data)
        assert errors == []

    def test_valid_hex_6_digits(self) -> None:
        data = {"name": "standard-hex", "colors": {"primary": "#aabbcc"}}
        errors = validate_theme(data)
        assert errors == []

    def test_valid_hex_8_digits(self) -> None:
        data = {"name": "alpha-hex", "colors": {"primary": "#aabbccdd"}}
        errors = validate_theme(data)
        assert errors == []

    def test_colors_not_a_dict_returns_error(self) -> None:
        data = {"name": "bad-colors", "colors": "not-a-dict"}
        errors = validate_theme(data)
        assert any("colors" in e.lower() for e in errors)

    def test_invalid_color_value_type(self) -> None:
        data = {
            "name": "bad-value-type",
            "colors": {
                "primary": [1, 2, 3],  # list is not allowed
            },
        }
        errors = validate_theme(data)
        assert len(errors) > 0

    def test_variables_not_a_dict_returns_error(self) -> None:
        data = {
            "name": "bad-vars",
            "colors": {},
            "variables": "not-a-dict",
        }
        errors = validate_theme(data)
        assert any("variables" in e.lower() for e in errors)

    def test_variable_value_not_string_returns_error(self) -> None:
        data = {
            "name": "bad-var-val",
            "colors": {},
            "variables": {"blue": 42},
        }
        errors = validate_theme(data)
        assert len(errors) > 0

    def test_not_a_dict_returns_error(self) -> None:
        errors = validate_theme("not a dict")  # type: ignore[arg-type]
        assert len(errors) > 0

    def test_color_value_integer_is_accepted(self) -> None:
        data = {
            "name": "int-color",
            "colors": {
                "primary": 255,
            },
        }
        errors = validate_theme(data)
        assert errors == []

    def test_valid_theme_with_variables(self) -> None:
        data = {
            "name": "with-vars",
            "variables": {"blue": "#0000ff"},
            "colors": {
                "primary": "blue",
            },
        }
        errors = validate_theme(data)
        assert errors == []

    def test_valid_theme_with_optional_fields(self) -> None:
        data = {
            "name": "full-theme",
            "description": "A complete theme",
            "author": "tester",
            "colors": {"primary": "#aabbcc"},
        }
        errors = validate_theme(data)
        assert errors == []


# ---------------------------------------------------------------------------
# TestThemeSchema
# ---------------------------------------------------------------------------


class TestThemeSchema:
    """Tests for the THEME_SCHEMA constant."""

    def test_schema_is_dict(self) -> None:
        assert isinstance(THEME_SCHEMA, dict)

    def test_schema_type_is_object(self) -> None:
        assert THEME_SCHEMA.get("type") == "object"

    def test_schema_has_properties(self) -> None:
        assert "properties" in THEME_SCHEMA

    def test_schema_requires_name_and_colors(self) -> None:
        required = THEME_SCHEMA.get("required", [])
        assert "name" in required
        assert "colors" in required

    def test_schema_defines_name_property(self) -> None:
        props = THEME_SCHEMA.get("properties", {})
        assert "name" in props

    def test_schema_defines_colors_property(self) -> None:
        props = THEME_SCHEMA.get("properties", {})
        assert "colors" in props
