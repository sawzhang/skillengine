# 05 — Structured output

Goal: get a validated Python object back from the agent, not a raw string.

## With a dataclass

```python
from dataclasses import dataclass

@dataclass
class TaskBreakdown:
    title: str
    steps: list[str]
    estimated_minutes: int

message, plan = await runner.chat_structured(
    "Break down 'ship the v1 release' into steps.",
    TaskBreakdown,
)
print(plan.title, plan.steps, plan.estimated_minutes)
```

## With a Pydantic model

```python
from pydantic import BaseModel, Field

class Review(BaseModel):
    summary: str = Field(..., description="One-paragraph summary.")
    rating: int = Field(..., ge=1, le=5)
    pros: list[str]
    cons: list[str]

_, review = await runner.chat_structured(
    "Review the book 'A Fire Upon the Deep'.",
    Review,
    max_retries=2,
)
```

Both Pydantic v1 and v2 are supported transparently.

## What happens on a bad reply

If the LLM returns prose, fenced code, or invalid JSON, SkillEngine:

1. Extracts the longest top-level JSON object/array from the reply.
2. Validates it against your type.
3. On failure, sends a correction prompt:
   > Your previous reply could not be parsed as the required JSON value: …

Up to `max_retries` corrections are attempted. After that,
`StructuredOutputError` is raised.

## Parsing without an agent

The same helpers are available standalone:

```python
from skillengine import extract_json_schema, parse_structured

schema = extract_json_schema(TaskBreakdown)   # JSON schema dict
plan = parse_structured(TaskBreakdown, llm_reply_text)
```
