"""Structured output for SkillEngine agents.

This module turns ``agent.chat_structured(user_input, output_type=...)`` into a
reliable round-trip: the LLM is instructed (via an appended system directive)
to emit a single JSON object that conforms to a schema derived from
``output_type``, the response is then extracted, parsed and validated.

Supported ``output_type`` shapes:

* **Pydantic v2 model** — ``BaseModel`` subclasses; schema via ``model_json_schema()``
* **Pydantic v1 model** — ``BaseModel`` subclasses; schema via ``schema()``
* **Python dataclass** — schema synthesised from ``__dataclass_fields__``
* **``TypedDict``** — schema synthesised from ``__annotations__``
* **Raw JSON schema** — passed in as a plain ``dict``

The module intentionally avoids any provider-specific JSON-mode flags so it
works with every existing :class:`~skillengine.adapters.base.LLMAdapter`.
Adapters may layer a faster strict-JSON path on top later without changing the
public API.
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StructuredOutputError(ValueError):
    """Raised when the LLM response cannot be coerced to ``output_type``."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_schema(tp: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a small JSON schema fragment.

    Handles the common cases (primitives, ``list[T]``, ``dict[str, T]``,
    ``Optional[T]``, ``Literal[...]``) and falls back to ``{}`` (any) for
    anything more exotic. The goal is *guidance for the LLM*, not validation,
    so coverage is intentionally pragmatic.
    """
    if tp is None or tp is type(None):
        return {"type": "null"}
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is None:
        if tp in _JSON_TYPE_MAP:
            return {"type": _JSON_TYPE_MAP[tp]}
        return {}
    if origin in (list, tuple):
        item_schema = _python_type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        value_schema = _python_type_to_schema(args[1]) if len(args) >= 2 else {}
        return {"type": "object", "additionalProperties": value_schema}
    if origin is typing.Union or origin is getattr(typing, "UnionType", object()):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_schema(non_none[0])
        return {"anyOf": [_python_type_to_schema(a) for a in non_none]}
    if origin is typing.Literal:
        return {"enum": list(args)}
    return {}


def extract_json_schema(output_type: Any) -> dict[str, Any]:
    """Return a JSON-schema dict describing ``output_type``."""
    # Raw dict — assume the caller already authored a JSON schema.
    if isinstance(output_type, dict):
        return dict(output_type)

    # Pydantic v2
    model_json_schema = getattr(output_type, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            return dict(model_json_schema())
        except Exception:  # pragma: no cover - defensive
            pass

    # Pydantic v1
    schema_fn = getattr(output_type, "schema", None)
    if callable(schema_fn) and not isinstance(output_type, type) and False:
        # Pydantic v1 instances expose ``schema`` as a classmethod too, but we
        # only want it when the *class* itself defines it.
        return dict(schema_fn())
    if isinstance(output_type, type) and callable(getattr(output_type, "schema", None)):
        try:
            return dict(output_type.schema())  # type: ignore[attr-defined]
        except Exception:
            pass

    # Dataclass
    if dataclasses.is_dataclass(output_type) and isinstance(output_type, type):
        # ``f.type`` is often a *string* (PEP 563 / ``from __future__ import
        # annotations``). Resolve to real types via ``get_type_hints``.
        try:
            hints = typing.get_type_hints(output_type)
        except Exception:
            hints = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for f in dataclasses.fields(output_type):
            tp = hints.get(f.name, f.type)
            properties[f.name] = _python_type_to_schema(tp)
            has_default = (
                f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            )
            if not has_default:
                required.append(f.name)
        schema: dict[str, Any] = {
            "type": "object",
            "title": output_type.__name__,
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    # TypedDict
    if isinstance(output_type, type) and issubclass_safe(output_type, dict):
        try:
            annotations = typing.get_type_hints(output_type)
        except Exception:
            annotations = getattr(output_type, "__annotations__", None) or {}
        if annotations:
            properties = {k: _python_type_to_schema(v) for k, v in annotations.items()}
            required_keys = list(getattr(output_type, "__required_keys__", annotations.keys()))
            return {
                "type": "object",
                "title": output_type.__name__,
                "properties": properties,
                "required": required_keys,
            }

    # Fallback — accept any object.
    return {"type": "object"}


def issubclass_safe(cls: Any, base: type) -> bool:
    try:
        return issubclass(cls, base)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(text: str) -> str:
    """Return the most likely JSON payload embedded in ``text``.

    Order of preference:

    1. The first fenced ``json`` block (``\\`\\`\\`json ... \\`\\`\\```).
    2. The longest balanced ``{...}`` or ``[...]`` substring.
    3. The original ``text`` (so callers see a useful error message).
    """
    if not text:
        return text
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)

    # Locate the outermost {...} or [...] payload via brace matching.
    best = _scan_balanced(text, "{", "}")
    alt = _scan_balanced(text, "[", "]")
    if best is None or (alt is not None and len(alt) > len(best)):
        best = alt
    return best if best is not None else text.strip()


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_structured(output_type: Any, text: str) -> Any:
    """Extract JSON from ``text`` and coerce to an instance of ``output_type``.

    Returns:
        * For Pydantic models — a validated model instance.
        * For dataclasses — an instance built from keyword arguments.
        * Otherwise — the parsed Python value (typically a ``dict``/``list``).

    Raises:
        StructuredOutputError: if no JSON could be located, the JSON is
            malformed, or it fails validation.
    """
    payload_text = extract_json_payload(text)
    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"Could not parse JSON from model response: {exc.msg}",
            raw=text,
        ) from exc

    # Raw JSON schema → return parsed value.
    if isinstance(output_type, dict):
        return parsed

    # Pydantic v2
    if hasattr(output_type, "model_validate"):
        try:
            return output_type.model_validate(parsed)
        except Exception as exc:
            raise StructuredOutputError(
                f"Validation failed for {output_type.__name__}: {exc}",
                raw=text,
            ) from exc

    # Pydantic v1
    if hasattr(output_type, "parse_obj"):
        try:
            return output_type.parse_obj(parsed)
        except Exception as exc:
            raise StructuredOutputError(
                f"Validation failed for {output_type.__name__}: {exc}",
                raw=text,
            ) from exc

    # Dataclass
    if dataclasses.is_dataclass(output_type) and isinstance(output_type, type):
        if not isinstance(parsed, dict):
            raise StructuredOutputError(
                f"Expected JSON object for dataclass {output_type.__name__}, "
                f"got {type(parsed).__name__}",
                raw=text,
            )
        try:
            return output_type(**parsed)
        except TypeError as exc:
            raise StructuredOutputError(
                f"Could not instantiate {output_type.__name__}: {exc}",
                raw=text,
            ) from exc

    # TypedDict / anything else — return the parsed object directly.
    return parsed


# ---------------------------------------------------------------------------
# Prompt directive
# ---------------------------------------------------------------------------


def build_directive(output_type: Any) -> str:
    """Build the system-prompt suffix instructing the model to emit JSON.

    The directive is intentionally short and provider-neutral. It contains:

    * a one-line imperative instruction,
    * the target JSON schema (inline, for the model to look at), and
    * an example response shell that pins the output format.
    """
    schema = extract_json_schema(output_type)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "\n\n## Structured output\n"
        "Your final reply MUST be a single JSON value that conforms to the "
        "schema below. Do not wrap it in prose, do not add commentary, do not "
        "use markdown fences. Emit only the JSON value.\n\n"
        f"```json-schema\n{schema_text}\n```\n"
    )
