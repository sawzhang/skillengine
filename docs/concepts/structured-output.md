# Structured output

`chat_structured()` returns both the assistant message and a parsed value of the
type you requested.

```python
from dataclasses import dataclass

@dataclass
class Plan:
    title: str
    steps: list[str]

message, plan = await runner.chat_structured("Plan a 3-step launch", Plan)
assert isinstance(plan, Plan)
```

## How it works

1. A JSON schema is extracted from `output_type` (Pydantic v1/v2, dataclass,
   `TypedDict`, or raw `dict`).
2. A `## Structured output` directive with the schema is appended to the system
   prompt for the duration of the call.
3. The LLM's reply is parsed: fenced code blocks are checked first, then the
   longest top-level `{...}` or `[...]` is balance-scanned out of the text.
4. The payload is validated against `output_type`. On failure, a correction
   prompt is sent (up to `max_retries` extra turns).

## Supported types

| Type | How it is validated |
|---|---|
| `pydantic.BaseModel` (v2) | `model_validate()` |
| `pydantic.BaseModel` (v1) | `parse_obj()` |
| `@dataclass` | constructed from the dict (PEP-563 type hints resolved) |
| `TypedDict` | required keys checked |
| `dict` | returned as-is |

## Helpers

- `extract_json_schema(T)` — schema for any supported `T`.
- `extract_json_payload(text)` — pulls a JSON value out of an arbitrary string.
- `parse_structured(T, text)` — combine the two and validate.
- `build_directive(T)` — the prompt suffix `chat_structured` appends.

All four are public on the top-level `skillengine` namespace.
