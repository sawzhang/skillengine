#!/usr/bin/env python3
"""
CodeModeRuntime Demo

Demonstrates the search + execute pattern inspired by Cloudflare's
code-mode-mcp. Instead of exposing N tools (one per API endpoint),
the runtime exposes just 2 tools — search and execute — and lets
the LLM write Python code against injected data and clients.

This example simulates an LLM agent discovering and calling a
REST API with 2500+ endpoints, using only ~1000 tokens of tool
definitions regardless of API size.

Usage:
    python examples/code_mode_demo.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skillengine.runtime.code_mode import CodeModeRuntime

# ── Fake OpenAPI spec (simulates a large API) ──────────────────────

OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Acme Cloud API", "version": "2.1.0"},
    "servers": [{"url": "https://api.acme.cloud/v2"}],
    "paths": {
        "/zones": {
            "get": {"summary": "List all zones", "tags": ["zones"]},
            "post": {"summary": "Create a zone", "tags": ["zones"]},
        },
        "/zones/{zone_id}": {
            "get": {"summary": "Get zone details", "tags": ["zones"]},
            "patch": {"summary": "Update zone settings", "tags": ["zones"]},
            "delete": {"summary": "Delete a zone", "tags": ["zones"]},
        },
        "/zones/{zone_id}/dns_records": {
            "get": {"summary": "List DNS records", "tags": ["dns"]},
            "post": {"summary": "Create DNS record", "tags": ["dns"]},
        },
        "/zones/{zone_id}/dns_records/{record_id}": {
            "get": {"summary": "Get DNS record", "tags": ["dns"]},
            "put": {"summary": "Update DNS record", "tags": ["dns"]},
            "delete": {"summary": "Delete DNS record", "tags": ["dns"]},
        },
        "/zones/{zone_id}/firewall/rules": {
            "get": {"summary": "List firewall rules", "tags": ["firewall"]},
            "post": {"summary": "Create firewall rule", "tags": ["firewall"]},
        },
        "/zones/{zone_id}/firewall/rules/{rule_id}": {
            "put": {"summary": "Update firewall rule", "tags": ["firewall"]},
            "delete": {"summary": "Delete firewall rule", "tags": ["firewall"]},
        },
        "/zones/{zone_id}/analytics/dashboard": {
            "get": {"summary": "Get analytics dashboard", "tags": ["analytics"]},
        },
        "/zones/{zone_id}/analytics/colos": {
            "get": {"summary": "Get analytics by colo", "tags": ["analytics"]},
        },
        "/user": {
            "get": {"summary": "Get current user", "tags": ["user"]},
            "patch": {"summary": "Update current user", "tags": ["user"]},
        },
        "/user/tokens": {
            "get": {"summary": "List API tokens", "tags": ["user"]},
            "post": {"summary": "Create API token", "tags": ["user"]},
        },
        "/accounts": {
            "get": {"summary": "List accounts", "tags": ["accounts"]},
        },
        "/accounts/{account_id}/workers/scripts": {
            "get": {"summary": "List Workers scripts", "tags": ["workers"]},
        },
        "/accounts/{account_id}/workers/scripts/{script_name}": {
            "get": {"summary": "Get Workers script", "tags": ["workers"]},
            "put": {"summary": "Upload Workers script", "tags": ["workers"]},
            "delete": {"summary": "Delete Workers script", "tags": ["workers"]},
        },
        "/accounts/{account_id}/r2/buckets": {
            "get": {"summary": "List R2 buckets", "tags": ["r2"]},
            "post": {"summary": "Create R2 bucket", "tags": ["r2"]},
        },
    },
}


# ── Fake API client ────────────────────────────────────────────────


class AcmeClient:
    """Simulates an authenticated API client."""

    def __init__(self, base_url: str, token: str = "sk-demo-xxx") -> None:
        self.base_url = base_url
        self.token = token
        self._db = {
            "zones": [
                {"id": "z1", "name": "example.com", "status": "active"},
                {"id": "z2", "name": "myapp.io", "status": "active"},
            ],
            "dns_records": [
                {"id": "r1", "zone_id": "z1", "type": "A", "name": "example.com", "content": "93.184.216.34"},
                {"id": "r2", "zone_id": "z1", "type": "CNAME", "name": "www", "content": "example.com"},
                {"id": "r3", "zone_id": "z2", "type": "A", "name": "myapp.io", "content": "10.0.0.1"},
            ],
        }

    def get(self, path: str) -> dict:
        """Simulate GET request."""
        if path == "/zones":
            return {"success": True, "result": self._db["zones"]}
        if path.startswith("/zones/") and path.endswith("/dns_records"):
            zone_id = path.split("/")[2]
            records = [r for r in self._db["dns_records"] if r["zone_id"] == zone_id]
            return {"success": True, "result": records}
        if path == "/user":
            return {"success": True, "result": {"name": "Demo User", "email": "demo@acme.cloud"}}
        return {"success": True, "result": [], "note": f"Simulated GET {path}"}

    def post(self, path: str, data: dict | None = None) -> dict:
        """Simulate POST request."""
        return {"success": True, "result": {"id": "new-123", **( data or {})}, "note": f"Simulated POST {path}"}


# ── Demo ───────────────────────────────────────────────────────────


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


async def main() -> None:
    print("=" * 60)
    print("  CodeModeRuntime Demo")
    print("  search + execute pattern for LLM agents")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────

    client = AcmeClient(base_url="https://api.acme.cloud/v2")

    runtime = CodeModeRuntime(
        spec=OPENAPI_SPEC,
        ctx={"client": client},
    )

    # ── 1. Tool definitions ────────────────────────────────────

    header("1. Tool Definitions (what the LLM sees)")

    tools = runtime.get_tool_definitions()
    print(f"Total tools: {len(tools)} (vs {sum(len(m) for m in OPENAPI_SPEC['paths'].values())} individual endpoints)")
    print()
    for tool in tools:
        fn = tool["function"]
        print(f"  tool: {fn['name']}")
        print(f"  desc: {fn['description'][:80]}...")
        print()

    # ── 2. Search: discover endpoints ──────────────────────────

    header("2. Search Phase — discover DNS endpoints")

    result = await runtime.search("""
# Find all DNS-related endpoints
endpoints = []
for path, methods in spec['paths'].items():
    for method, op in methods.items():
        if 'dns' in op.get('tags', []):
            endpoints.append({
                'method': method.upper(),
                'path': path,
                'summary': op['summary']
            })
result = endpoints
""")

    print(f"  success: {result.success}")
    print(f"  LLM code found {len(json.loads(result.output))} DNS endpoints:")
    for ep in json.loads(result.output):
        print(f"    {ep['method']:6s} {ep['path']:45s} — {ep['summary']}")

    # ── 3. Search: filter by tag ───────────────────────────────

    header("3. Search Phase — list all available tags")

    result = await runtime.search("""
tags = set()
for path, methods in spec['paths'].items():
    for method, op in methods.items():
        for tag in op.get('tags', []):
            tags.add(tag)
result = sorted(tags)
""")

    print(f"  Available tags: {json.loads(result.output)}")

    # ── 4. Execute: call the API ───────────────────────────────

    header("4. Execute Phase — list zones via API client")

    result = await runtime.run("""
# Use the injected client to call the API
response = ctx['client'].get('/zones')
result = {
    'success': response['success'],
    'zones': [{'name': z['name'], 'status': z['status']} for z in response['result']]
}
""")

    print(f"  success: {result.success}")
    data = json.loads(result.output)
    print(f"  API returned {len(data['zones'])} zones:")
    for zone in data["zones"]:
        print(f"    {zone['name']} ({zone['status']})")

    # ── 5. Execute: chain search → execute ─────────────────────

    header("5. Chained — search spec then call API")

    result = await runtime.run("""
# Step 1: Find the DNS list endpoint from the spec
dns_list_path = None
for path, methods in spec['paths'].items():
    if 'get' in methods and 'dns_records' in path and '{record_id}' not in path:
        dns_list_path = path
        break

# Step 2: Call it for zone z1
if dns_list_path:
    actual_path = dns_list_path.replace('{zone_id}', 'z1')
    response = ctx['client'].get(actual_path)
    result = {
        'endpoint_from_spec': dns_list_path,
        'actual_path': actual_path,
        'records': response['result']
    }
else:
    result = {'error': 'DNS endpoint not found in spec'}
""")

    print(f"  success: {result.success}")
    data = json.loads(result.output)
    print(f"  Discovered endpoint: {data['endpoint_from_spec']}")
    print(f"  Called: {data['actual_path']}")
    print(f"  DNS records for zone z1:")
    for rec in data["records"]:
        print(f"    {rec['type']:5s} {rec['name']:20s} → {rec['content']}")

    # ── 6. Engine integration ──────────────────────────────────

    header("6. Engine Integration — drop-in replacement for BashRuntime")

    from skillengine import SkillsConfig, SkillsEngine

    engine = SkillsEngine(
        config=SkillsConfig(skill_dirs=[]),
        runtime=runtime,
    )

    # Engine.execute now runs Python code instead of bash commands
    result = await engine.execute(
        "result = {'api': spec['info']['title'], 'version': spec['info']['version']}"
    )
    data = json.loads(result.output)
    print(f"  Engine executed code-mode: {data}")

    # ── 7. Subprocess isolation ────────────────────────────────

    header("7. Subprocess Sandbox — isolated execution")

    sandboxed = CodeModeRuntime(
        spec=OPENAPI_SPEC,
        ctx={"greeting": "hello from sandbox"},
        sandbox="subprocess",
    )

    result = await sandboxed.search("""
result = {
    'path_count': len(spec['paths']),
    'api_title': spec['info']['title'],
}
""")

    print(f"  Subprocess search success: {result.success}")
    print(f"  Result: {json.loads(result.output)}")

    result = await sandboxed.run("""
result = f"{ctx['greeting']} — API has {len(spec['paths'])} paths"
""")
    print(f"  Subprocess execute: {result.output}")

    # ── Summary ────────────────────────────────────────────────

    header("Summary")

    total_endpoints = sum(len(m) for m in OPENAPI_SPEC["paths"].values())
    print(f"  API endpoints:    {total_endpoints}")
    print(f"  Tools exposed:    {len(tools)} (search + execute)")
    print(f"  Token reduction:  O({total_endpoints}) → O(1)")
    print()
    print("  The LLM writes Python code to discover and call any")
    print("  endpoint, without needing a dedicated tool for each one.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
