# Interop

SkillEngine speaks three agent ecosystems out of the box.

## MCP (Model Context Protocol)

**As a client** — call any MCP server's tools as if they were local:

```python
from skillengine import MCPServerSpec
pool = await runner.connect_mcp_servers([
    MCPServerSpec(command="npx", args=["-y", "@modelcontextprotocol/server-everything"]),
])
# Tools appear in the dispatcher as "<name>__<tool>".
```

**As a server** — expose your skills/tools to Claude Desktop / Cursor / Cline:

```bash
python -m skillengine.mcp --skill-dir ./skills --name my-skills
```

The CLI speaks JSON-RPC 2.0 over stdio. Skills become `skill__<name>` tools.

**Spec URIs**:

| Form | Example |
|---|---|
| Shell | `mcp+stdio:npx -y @mcp/server-everything` |
| URL | `mcp+stdio://node?args=server.js&env=DEBUG=1` |
| JSON | `mcp+command:{"command":"node","args":["s.js"]}` |

Use `skillengine.mcp.parse_mcp_uri()` to convert any of these into an
`MCPServerSpec`.

## A2A (Agent-to-Agent)

The `skillengine.a2a` package implements the A2A draft protocol with:

- `AgentCard` derived automatically from your skills
- `A2AClient` for sending tasks to remote agents
- `AgentRegistry` + `AgentDiscovery` for service discovery
- `PerformanceRouter` for picking the fastest healthy agent

## Handoffs (OpenAI Agents SDK / Anthropic A2A compat)

A handoff is a synthetic `transfer_to_<agent>` tool that delegates to another
agent. SkillEngine ships factories for the three common targets:

```python
from skillengine import handoff, callable_handoff, agent_handoff, a2a_handoff

h1 = callable_handoff(my_fn, name="formatter")
h2 = agent_handoff(child_runner, name="researcher")
h3 = a2a_handoff(client, endpoint="http://remote", skill_name="analyze",
                 name="analyzer")
runner.add_handoffs([h1, h2, h3])
```

An optional `input_filter` lets you trim the message history before transfer,
matching the OpenAI Agents SDK extension point.
