"""A2A Coordinator Demo — multi-agent routing through a central coordinator.

Architecture::

    External Caller (this script)
         │  POST /tasks  (to coordinator)
         ▼
    CoordinatorAgent  (port 8000)
    │  discovers + routes via A2AClient
    ├── WeatherAgent   (port 8001)  — weather forecasts & climate queries
    ├── CodeAgent      (port 8002)  — code review, debugging, refactoring
    └── MathAgent      (port 8003)  — calculations, equations, statistics

Run::

    uv run python examples/a2a_coordinator_demo.py

The demo:
1. Starts 3 mock worker-agent servers (no LLM — deterministic responses)
2. Starts the coordinator on port 8000 and connects it to all 3
3. Sends 6 test queries to the coordinator
4. Shows which agent handled each request + routing scores
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any

import httpx

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coordinator_demo")

# ── Worker agent definitions ──────────────────────────────────────────────────
# Each agent is a tiny FastAPI app that returns deterministic responses.
# Real deployments would use A2AServer wrapping a SkillsEngine.

WORKER_AGENTS: list[dict[str, Any]] = [
    {
        "name": "weather-agent",
        "port": 8001,
        "description": (
            "Provides weather forecasts, climate data, and meteorological analysis. "
            "Use for questions about temperature, rain, humidity, and seasonal trends."
        ),
        "version": "1.0.0",
        "tags": ["weather", "forecast", "climate", "temperature", "meteorology"],
        "cost_hint": "low",
        "handler": lambda text: (
            f"[Weather] Based on your query '{text}': "
            "Current conditions show partly cloudy skies with a high of 22°C. "
            "Wind from the northwest at 15 km/h. 30% chance of rain this afternoon."
        ),
    },
    {
        "name": "code-agent",
        "port": 8002,
        "description": (
            "Expert in code review, debugging, refactoring, and software architecture. "
            "Handles Python, JavaScript, TypeScript, Go, and Rust. "
            "Use for code quality, bug fixes, and best practices."
        ),
        "version": "1.0.0",
        "tags": [
            "code", "review", "debug", "refactor",
            "python", "javascript", "programming", "software",
        ],
        "cost_hint": "medium",
        "handler": lambda text: (
            f"[CodeAgent] Analysis of '{text[:50]}...': "
            "Code review complete. Found 2 issues: "
            "(1) Missing error handling in the HTTP client — add try/except for httpx.HTTPError. "
            "(2) Variable 'data' shadows built-in — rename to 'response_data'. "
            "Overall code quality: B+. Recommend adding type hints."
        ),
    },
    {
        "name": "math-agent",
        "port": 8003,
        "description": (
            "Solves mathematical problems including arithmetic, algebra, calculus, "
            "statistics, and linear algebra. Provides step-by-step solutions."
        ),
        "version": "1.0.0",
        "tags": ["math", "calculation", "statistics", "algebra", "calculus", "equation", "formula"],
        "cost_hint": "low",
        "handler": lambda text: (
            f"[MathAgent] Computing '{text}': "
            "Step 1: Parse expression. "
            "Step 2: Apply order of operations. "
            "Step 3: Simplify. "
            "Result = 42 (the answer to everything). "
            "Confidence: 100%."
        ),
    },
]

# ── Worker agent server factory ───────────────────────────────────────────────

def make_worker_app(agent_def: dict[str, Any]) -> Any:
    """Create a FastAPI app for a mock worker agent."""
    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError("fastapi required: pip install fastapi uvicorn")

    app = FastAPI(title=agent_def["name"])

    @app.get("/.well-known/agent.json")
    async def agent_card() -> JSONResponse:
        card = {
            "server": agent_def["name"],
            "version": agent_def["version"],
            "agents": [
                {
                    "name": agent_def["name"],
                    "description": agent_def["description"],
                    "version": agent_def["version"],
                    "tags": agent_def["tags"],
                    "cost_hint": agent_def["cost_hint"],
                    "capabilities": {
                        "streaming": False,
                        "multi_turn": False,
                        "push_notifications": False,
                    },
                    "skills": [],
                }
            ],
        }
        return JSONResponse(content=card)

    @app.post("/tasks")
    async def create_task(request: dict[str, Any]) -> JSONResponse:
        import uuid

        task_id = request.get("task_id", uuid.uuid4().hex[:12])
        input_text = request.get("input_text", "")

        # Simulate light processing time
        await asyncio.sleep(0.05)

        output = agent_def["handler"](input_text)
        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "completed",
                "output": output,
                "metadata": {
                    "handled_by": agent_def["name"],
                    "agent_port": agent_def["port"],
                },
            }
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "agent": agent_def["name"]}

    return app


# ── Server launcher helpers ───────────────────────────────────────────────────

def _run_uvicorn(app: Any, host: str, port: int) -> None:
    """Run uvicorn in a background thread."""
    import uvicorn  # type: ignore[import-not-found]

    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_worker_servers() -> list[threading.Thread]:
    """Start all worker agent servers in background threads."""
    threads = []
    for agent_def in WORKER_AGENTS:
        app = make_worker_app(agent_def)
        t = threading.Thread(
            target=_run_uvicorn,
            args=(app, "127.0.0.1", agent_def["port"]),
            daemon=True,
            name=f"worker-{agent_def['name']}",
        )
        t.start()
        threads.append(t)
        logger.info("Started worker server: %s on port %d", agent_def["name"], agent_def["port"])

    return threads


def start_coordinator_server(coordinator: Any, port: int = 8000) -> threading.Thread:
    """Start the coordinator server in a background thread."""
    import uvicorn  # type: ignore[import-not-found]

    app = coordinator.create_app(base_url=f"http://127.0.0.1:{port}")

    t = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True,
        name="coordinator-server",
    )
    t.start()
    logger.info("Started coordinator on port %d", port)
    return t


async def wait_for_server(url: str, retries: int = 30, delay: float = 0.5) -> bool:
    """Poll a server URL until it responds (async-safe)."""
    async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
        for _ in range(retries):
            try:
                await client.get(url)
                return True
            except Exception:
                await asyncio.sleep(delay)
    return False


# ── Demo queries ──────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # (skill_name_hint, user_query)
    ("auto", "What will the weather be like in Tokyo tomorrow?"),
    ("auto", "Review this Python function for code quality issues"),
    ("auto", "Statistics calculation: standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]"),
    ("auto", "Weather forecast: will it rain this weekend in London?"),
    ("auto", "Help me debug this JavaScript TypeError in my async function"),
    # Direct routing by exact agent name
    ("math-agent", "Solve the quadratic equation x² + 5x + 6 = 0"),
]


async def send_query(
    coordinator_url: str,
    skill_name: str,
    input_text: str,
) -> dict[str, Any]:
    """Send a task to the coordinator and return the response."""
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        resp = await client.post(
            f"{coordinator_url}/tasks",
            json={
                "skill_name": skill_name,
                "input_text": input_text,
            },
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]


async def fetch_agents(coordinator_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        resp = await client.get(f"{coordinator_url}/agents")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]


async def fetch_health(coordinator_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        resp = await client.get(f"{coordinator_url}/health")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]


# ── Main demo ─────────────────────────────────────────────────────────────────

async def run_demo() -> None:
    """Full coordinator demo: start servers, connect agents, route queries."""
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        print("ERROR: fastapi and uvicorn are required.")
        print("Install with:  pip install fastapi uvicorn")
        sys.exit(1)

    coordinator_url = "http://127.0.0.1:8000"

    # ── 1. Start worker agents ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("A2A COORDINATOR DEMO")
    print("=" * 70)
    print("\n[1/4] Starting worker agent servers...")

    start_worker_servers()

    # Wait for all workers to be ready
    for agent_def in WORKER_AGENTS:
        url = f"http://127.0.0.1:{agent_def['port']}/health"
        if await wait_for_server(url):
            print(f"      ✓ {agent_def['name']}  (port {agent_def['port']})")
        else:
            print(f"      ✗ {agent_def['name']} — timed out!")
            sys.exit(1)

    # ── 2. Create and connect coordinator ─────────────────────────────────
    print("\n[2/4] Starting coordinator...")

    from skillengine.a2a.coordinator import CoordinatorAgent, CoordinatorConfig

    config = CoordinatorConfig(
        name="demo-coordinator",
        description=(
            "Central demo coordinator that routes tasks to weather, code, and math agents."
        ),
        version="0.1.0",
        tags=["coordinator", "demo"],
        remote_endpoints=[
            f"http://127.0.0.1:{a['port']}" for a in WORKER_AGENTS
        ],
        connect_timeout=5.0,
    )
    coordinator = CoordinatorAgent(config)

    # Connect to all downstream agents BEFORE starting the server
    print("      Discovering downstream agents...")
    results = await coordinator.connect_all()
    for endpoint, names in results.items():
        status = "✓" if names else "✗"
        print(f"      {status} {endpoint}  →  {names}")

    # Start coordinator HTTP server
    start_coordinator_server(coordinator, port=8000)
    if not await wait_for_server(f"{coordinator_url}/health"):
        print("      ✗ Coordinator timed out!")
        sys.exit(1)
    print(f"      ✓ Coordinator running at {coordinator_url}")

    # ── 3. Show connected agents ──────────────────────────────────────────
    print("\n[3/4] Connected agent registry:")
    agents_info = await fetch_agents(coordinator_url)
    for agent in agents_info["agents"]:
        print(f"      • {agent['name']:<20}  {agent['description'][:55]}…")
        print(f"        tags: {agent['tags']}")

    # Also fetch coordinator's own agent card
    async with httpx.AsyncClient(trust_env=False) as c:
        card_resp = await c.get(f"{coordinator_url}/.well-known/agent.json")
        card = card_resp.json()
    print(f"\n      Coordinator card: {card['name']} v{card['version']}")
    print(f"      Aggregated skills: {[s['name'] for s in card.get('skills', [])]}")

    # ── 4. Route test queries through coordinator ─────────────────────────
    print("\n[4/4] Routing test queries through coordinator:")
    print("-" * 70)

    for skill_hint, query in TEST_QUERIES:
        routing_label = f"skill='{skill_hint}'" if skill_hint != "auto" else "auto-route"
        print(f"\n  Query [{routing_label}]:")
        print(f"  \"{query}\"")

        try:
            response = await send_query(coordinator_url, skill_hint, query)
            routed_to = response.get("metadata", {}).get("handled_by", "?")
            status = response.get("status", "?")
            output = response.get("output", "")[:120]
            print(f"  → Status:    {status}")
            print(f"  → Routed to: {routed_to}")
            print(f"  → Output:    {output}...")

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            print(f"  → HTTP {e.response.status_code}: {error_body}")
        except Exception as e:
            print(f"  → Error: {e}")

    # ── Final health check ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("\nFinal coordinator health:")
    health = await fetch_health(coordinator_url)
    print(f"  Agents connected: {health['agents_connected']}")
    for agent in health["agents"]:
        print(
            f"  • {agent['name']:<20}  calls={agent['calls']}  "
            f"success_rate={agent['success_rate']:.0%}  "
            f"avg_latency={agent['avg_latency_ms']:.0f}ms"
        )

    print("\n" + "=" * 70)
    print("Demo complete.")
    print()
    print("Coordinator API:")
    print(f"  GET  {coordinator_url}/.well-known/agent.json")
    print(f"  POST {coordinator_url}/tasks")
    print(f"  GET  {coordinator_url}/agents")
    print(f"  GET  {coordinator_url}/health")
    print(f"  POST {coordinator_url}/agents/connect")
    print()
    print("Press Ctrl+C to stop all servers.")
    print("=" * 70 + "\n")

    # Keep running so you can send manual requests
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    asyncio.run(run_demo())
