# 04 — Multi-agent handoffs

Goal: route a conversation to a specialised sub-agent on demand, using the same
`transfer_to_<agent>` pattern as the OpenAI Agents SDK.

## Three flavours

```python
from skillengine import (
    a2a_handoff,
    agent_handoff,
    callable_handoff,
)

# (a) Wrap a plain function.
formatter = callable_handoff(
    lambda text: text.upper(),
    name="formatter",
    description="Shout the text in caps.",
)

# (b) Wrap a local AgentRunner.
researcher = agent_handoff(
    child_runner,
    name="researcher",
    description="Deep-research questions about science papers.",
)

# (c) Wrap a remote A2A agent.
from skillengine.a2a import A2AClient
analyzer = a2a_handoff(
    A2AClient(),
    endpoint="http://analyzer.internal:8080",
    skill_name="analyze",
    name="analyzer",
)
```

## Register and chat

```python
runner.add_handoffs([formatter, researcher, analyzer])
reply = await runner.chat(
    "Research the latest on quantum error correction, then format the summary."
)
print(reply.content)
```

The LLM now has three extra tools: `transfer_to_formatter`,
`transfer_to_researcher`, `transfer_to_analyzer`. Each one forwards its `input`
argument to the target and returns the target's text.

## Filtering history before transfer

Match the OpenAI SDK's `handoff(input_filter=...)` extension point:

```python
from skillengine import handoff

def last_message_only(messages: list[dict]) -> list[dict]:
    return messages[-1:]

async def target(text: str, ctx: dict) -> str:
    # ctx["messages"] will contain only the last message.
    return "ok"

h = handoff(target, name="summarizer", input_filter=last_message_only)
```

## Clean up

```python
runner.remove_handoffs()   # unregisters every handoff added via add_handoffs()
```
