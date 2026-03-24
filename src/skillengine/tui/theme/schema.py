"""Theme JSON schema validation."""

from __future__ import annotations

from typing import Any

THEME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "author": {"type": "string"},
        "variables": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "colors": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
            },
        },
    },
    "required": ["name", "colors"],
}


def validate_theme(data: dict[str, Any]) -> list[str]:
    """Validate a theme data dict against the schema.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Theme must be a JSON object"]

    if "name" not in data:
        errors.append("Missing required field: 'name'")

    if "colors" not in data:
        errors.append("Missing required field: 'colors'")
    elif not isinstance(data["colors"], dict):
        errors.append("'colors' must be an object")
    else:
        for key, value in data["colors"].items():
            if not isinstance(key, str):
                errors.append(f"Color key must be a string, got: {type(key).__name__}")
            if not isinstance(value, (str, int)):
                errors.append(
                    f"Color value for '{key}' must be a string or integer, "
                    f"got: {type(value).__name__}"
                )
            if isinstance(value, str) and value.startswith("#"):
                # Validate hex color
                hex_part = value[1:]
                if len(hex_part) not in (3, 6, 8):
                    errors.append(f"Invalid hex color for '{key}': {value}")
                elif not all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    errors.append(f"Invalid hex color for '{key}': {value}")

    if "variables" in data:
        if not isinstance(data["variables"], dict):
            errors.append("'variables' must be an object")
        else:
            for key, value in data["variables"].items():
                if not isinstance(value, str):
                    errors.append(f"Variable '{key}' must be a string, got: {type(value).__name__}")

    return errors
