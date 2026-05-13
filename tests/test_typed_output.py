"""Tests for the structured output (``output_type``) pipeline."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

from skillengine.adapters.base import AgentResponse, LLMAdapter, Message
from skillengine.adapters.registry import AdapterRegistry
from skillengine.agent import AgentConfig, AgentRunner
from skillengine.engine import SkillsEngine
from skillengine.events import EventBus
from skillengine.typed_output import (
    StructuredOutputError,
    build_directive,
    extract_json_payload,
    extract_json_schema,
    parse_structured,
)

# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Person:
    name: str
    age: int


@dataclasses.dataclass
class PersonWithDefault:
    name: str
    age: int = 0


def test_extract_schema_from_dataclass() -> None:
    schema = extract_json_schema(Person)
    assert schema["type"] == "object"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert set(schema["required"]) == {"name", "age"}


def test_extract_schema_dataclass_with_default_excludes_from_required() -> None:
    schema = extract_json_schema(PersonWithDefault)
    assert schema["required"] == ["name"]


def test_extract_schema_passes_raw_dict_through() -> None:
    custom = {"type": "object", "properties": {"x": {"type": "string"}}}
    out = extract_json_schema(custom)
    assert out == custom
    # must be a *copy*, not the same object — callers may mutate.
    assert out is not custom


def test_extract_schema_from_typed_dict() -> None:
    from typing import TypedDict

    class Pet(TypedDict):
        name: str
        legs: int

    schema = extract_json_schema(Pet)
    assert schema["type"] == "object"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["legs"]["type"] == "integer"
    assert set(schema["required"]) == {"name", "legs"}


def test_extract_schema_pydantic_v2() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        qty: int = 1

    schema = extract_json_schema(Item)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "qty" in schema["properties"]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_payload_plain_object() -> None:
    assert extract_json_payload('{"a": 1}') == '{"a": 1}'


def test_extract_payload_with_prose() -> None:
    text = 'Sure! Here you go:\n\n{"name": "Ada", "age": 36}\n\nLet me know if...'
    assert extract_json_payload(text) == '{"name": "Ada", "age": 36}'


def test_extract_payload_fenced_block() -> None:
    text = 'Here is the data:\n```json\n{"a": 1, "b": [1, 2]}\n```\nDone.'
    assert extract_json_payload(text) == '{"a": 1, "b": [1, 2]}'


def test_extract_payload_array() -> None:
    text = "before [1, 2, 3] after"
    assert extract_json_payload(text) == "[1, 2, 3]"


def test_extract_payload_prefers_longer_payload() -> None:
    # Array is longer than the embedded object — should win.
    text = '{"x": 1} and the real data is [1, 2, 3, 4, 5, 6]'
    assert extract_json_payload(text) == "[1, 2, 3, 4, 5, 6]"


def test_extract_payload_nested_braces() -> None:
    text = '{"outer": {"inner": {"value": 42}}}'
    assert extract_json_payload(text) == text


def test_extract_payload_handles_braces_inside_strings() -> None:
    text = '{"text": "this has a } in it", "ok": true}'
    assert extract_json_payload(text) == text


def test_extract_payload_no_json_returns_input() -> None:
    text = "no json here"
    assert extract_json_payload(text) == "no json here"


# ---------------------------------------------------------------------------
# parse_structured
# ---------------------------------------------------------------------------


def test_parse_structured_dataclass() -> None:
    result = parse_structured(Person, '{"name": "Ada", "age": 36}')
    assert isinstance(result, Person)
    assert result.name == "Ada"
    assert result.age == 36


def test_parse_structured_dataclass_from_prose() -> None:
    text = 'Of course! ```json\n{"name": "Ada", "age": 36}\n``` Hope this helps.'
    result = parse_structured(Person, text)
    assert isinstance(result, Person)
    assert result.name == "Ada"


def test_parse_structured_raw_schema_returns_value() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    result = parse_structured(schema, '{"x": 7}')
    assert result == {"x": 7}


def test_parse_structured_typed_dict_returns_dict() -> None:
    from typing import TypedDict

    class P(TypedDict):
        a: int

    result = parse_structured(P, '{"a": 5}')
    assert result == {"a": 5}


def test_parse_structured_pydantic_v2() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        qty: int

    result = parse_structured(Item, '{"name": "ball", "qty": 3}')
    assert result.name == "ball"
    assert result.qty == 3


def test_parse_structured_invalid_json_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured(Person, "this is not json at all")


def test_parse_structured_dataclass_wrong_shape() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured(Person, '{"wrong_field": 1}')


def test_parse_structured_dataclass_requires_object() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured(Person, "[1, 2, 3]")


def test_parse_structured_pydantic_validation_error() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Strict(BaseModel):
        n: int

    with pytest.raises(StructuredOutputError):
        parse_structured(Strict, '{"n": "not a number"}')


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------


def test_build_directive_includes_schema_json() -> None:
    directive = build_directive(Person)
    assert "Structured output" in directive
    assert '"type": "object"' in directive
    assert '"name"' in directive
    assert '"age"' in directive


# ---------------------------------------------------------------------------
# AgentRunner.chat_structured — integration with a fake adapter
# ---------------------------------------------------------------------------


class _ScriptedAdapter(LLMAdapter):
    """LLMAdapter that returns scripted responses from a queue."""

    def __init__(self, engine: SkillsEngine, responses: list[str]) -> None:
        super().__init__(engine)
        self.responses = list(responses)
        self.system_prompts: list[str | None] = []

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        self.system_prompts.append(system_prompt)
        if not self.responses:
            return AgentResponse(content="", finish_reason="stop")
        content = self.responses.pop(0)
        return AgentResponse(content=content, finish_reason="stop")


def _make_runner(responses: list[str]) -> tuple[AgentRunner, _ScriptedAdapter]:
    engine = MagicMock(spec=SkillsEngine)
    engine.get_snapshot.return_value = MagicMock(skills=[], prompt="", get_skill=lambda n: None)
    config = AgentConfig(
        model="test-model",
        base_url="http://localhost",
        api_key="test-key",
        max_turns=2,
        enable_tools=False,
        auto_execute=False,
    )
    registry = AdapterRegistry()
    adapter = _ScriptedAdapter(engine, responses)
    registry.register("scripted", adapter)
    registry.set_default("scripted")
    runner = AgentRunner(
        engine=engine,
        config=config,
        events=EventBus(),
        adapter_registry=registry,
    )
    runner._active_adapter_name = "scripted"
    return runner, adapter


@pytest.mark.asyncio
async def test_chat_structured_parses_dataclass() -> None:
    runner, adapter = _make_runner(['{"name": "Ada", "age": 36}'])
    message, value = await runner.chat_structured("introduce Ada", Person)
    assert isinstance(value, Person)
    assert value.name == "Ada"
    assert value.age == 36
    # The structured directive must have flowed into the system prompt.
    assert any("Structured output" in (sp or "") for sp in adapter.system_prompts)


@pytest.mark.asyncio
async def test_chat_structured_directive_cleared_after_call() -> None:
    runner, _adapter = _make_runner(['{"name": "Ada", "age": 36}'])
    await runner.chat_structured("hi", Person)
    assert runner._structured_directive is None


@pytest.mark.asyncio
async def test_chat_structured_retries_on_invalid_json() -> None:
    # First reply is unparseable, second reply is good. With max_retries=1
    # the agent gets exactly one correction round.
    runner, adapter = _make_runner(["definitely not json", '{"name": "Bob", "age": 21}'])
    _, value = await runner.chat_structured("introduce", Person, max_retries=1)
    assert isinstance(value, Person)
    assert value.name == "Bob"
    assert len(adapter.system_prompts) == 2


@pytest.mark.asyncio
async def test_chat_structured_raises_when_retries_exhausted() -> None:
    runner, _adapter = _make_runner(["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        await runner.chat_structured("introduce", Person, max_retries=1)


@pytest.mark.asyncio
async def test_chat_structured_with_raw_schema_returns_dict() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    runner, _adapter = _make_runner(['{"x": 11}'])
    _, value = await runner.chat_structured("compute", schema)
    assert value == {"x": 11}


@pytest.mark.asyncio
async def test_chat_structured_with_pydantic_model() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Order(BaseModel):
        item: str
        qty: int

    runner, _adapter = _make_runner(['{"item": "ball", "qty": 3}'])
    _, value = await runner.chat_structured("place order", Order)
    assert value.item == "ball"
    assert value.qty == 3


@pytest.mark.asyncio
async def test_chat_structured_directive_not_persisted_across_chats() -> None:
    runner, adapter = _make_runner(['{"name": "Ada", "age": 36}', "plain reply"])
    await runner.chat_structured("structured turn", Person)
    # Now call a plain chat — the directive should NOT be present.
    await runner.chat("plain turn")
    assert "Structured output" not in (adapter.system_prompts[-1] or "")
