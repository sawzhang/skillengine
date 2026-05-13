# 01 — Your first agent

Goal: run a single-turn agent against any OpenAI-compatible endpoint.

## Install

```bash
uv add skillengine
# or
pip install skillengine
```

## Code

```python
# first_agent.py
import asyncio
import os

from skillengine import (
    AgentConfig,
    AgentRunner,
    SkillsConfig,
    SkillsEngine,
)


async def main() -> None:
    engine = SkillsEngine(SkillsConfig(skill_dirs=[]))  # no skills yet
    runner = AgentRunner(
        engine=engine,
        config=AgentConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            max_turns=4,
        ),
    )
    reply = await runner.chat("Give me three creative dinner ideas.")
    print(reply.content)


asyncio.run(main())
```

## Run

```bash
export OPENAI_API_KEY=sk-...
python first_agent.py
```

That is the minimum surface. Everything else in the cookbook builds on it.
