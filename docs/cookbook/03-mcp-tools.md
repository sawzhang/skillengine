# 03 — Connect an MCP server

Goal: let the agent use tools provided by an external MCP server (e.g. the
reference everything-server).

## Code

```python
from skillengine import (
    AgentConfig,
    AgentRunner,
    MCPServerSpec,
    SkillsConfig,
    SkillsEngine,
)


async def main() -> None:
    runner = AgentRunner(
        engine=SkillsEngine(SkillsConfig(skill_dirs=[])),
        config=AgentConfig(model="gpt-4o-mini", base_url="...", api_key="..."),
    )
    await runner.connect_mcp_servers([
        MCPServerSpec(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            name="everything",
        ),
    ])
    try:
        reply = await runner.chat("List all tools you have available.")
        print(reply.content)
    finally:
        await runner.disconnect_mcp_servers()
```

## How tools are named

If the remote server exposes a tool called `echo`, it shows up to the LLM as
`everything__echo`. The prefix is the logical name from `MCPServerSpec.name` (or
the command basename if you omit it).

## URI form

```python
from skillengine import parse_mcp_uri
spec = parse_mcp_uri("mcp+stdio:npx -y @modelcontextprotocol/server-everything")
```

Useful when specs come from config files or environment variables.

## Running SkillEngine as an MCP server

The inverse direction — exposing your skills to Claude Desktop or Cursor:

```bash
python -m skillengine.mcp --skill-dir ./skills --name my-skills
```

Validate with the official inspector:

```bash
npx @modelcontextprotocol/inspector stdio python -m skillengine.mcp --skill-dir ./skills
```
