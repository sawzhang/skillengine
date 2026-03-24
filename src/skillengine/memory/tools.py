"""Tool definitions for LLM-callable memory operations."""

from __future__ import annotations

from typing import Any

from skillengine.extensions.models import ToolInfo
from skillengine.logging import get_logger
from skillengine.memory.client import OpenVikingClient

logger = get_logger("memory.tools")


class MemoryState:
    """Shared mutable state for memory tools (session ID, etc.)."""

    def __init__(self, client: OpenVikingClient) -> None:
        self.client = client
        self.session_id: str | None = None


def _format_results(results: list[dict[str, Any]]) -> str:
    """Format search results into a human-readable string."""
    if not results:
        return "[No memories found]"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        content = r.get("content", r.get("text", ""))
        score = r.get("score", "")
        uri = r.get("uri", "")
        header = f"{i}."
        if uri:
            header += f" [{uri}]"
        if score:
            header += f" (score: {score:.2f})" if isinstance(score, float) else f" (score: {score})"
        lines.append(header)
        lines.append(f"   {content}")
    return "\n".join(lines)


def make_recall_handler(state: MemoryState) -> Any:
    """Create handler for the ``recall_memory`` tool."""

    async def recall_memory(
        query: str,
        scope: str = "user",
        limit: int = 5,
    ) -> str:
        if not state.client.available:
            return "[Memory unavailable]"

        target_uri = f"viking://{scope}/memories/"

        if state.session_id:
            results = await state.client.search(
                query=query,
                target_uri=target_uri,
                session_id=state.session_id,
                limit=limit,
            )
        else:
            results = await state.client.find(
                query=query,
                target_uri=target_uri,
                limit=limit,
            )

        if results is None:
            return "[Memory unavailable]"
        return _format_results(results)

    return recall_memory


def make_save_handler(state: MemoryState) -> Any:
    """Create handler for the ``save_memory`` tool."""

    async def save_memory(
        content: str,
        category: str = "preferences",
    ) -> str:
        if not state.client.available:
            return "[Memory unavailable]"
        if not state.session_id:
            return "[No active memory session]"

        annotated = f"[memory:{category}] {content}"
        ok = await state.client.add_message(state.session_id, "assistant", annotated)
        if not ok:
            return "[Failed to save memory]"

        committed = await state.client.commit_session(state.session_id)
        if committed:
            return f"Memory saved ({category}): {content[:80]}"
        return f"Memory saved ({category}, commit pending): {content[:80]}"

    return save_memory


def make_explore_handler(state: MemoryState) -> Any:
    """Create handler for the ``explore_memory`` tool."""

    async def explore_memory(
        uri: str = "viking://user/memories/",
        recursive: bool = False,
    ) -> str:
        if not state.client.available:
            return "[Memory unavailable]"

        entries = await state.client.ls(uri=uri, recursive=recursive)
        if entries is None:
            return "[Memory unavailable]"
        if not entries:
            return "[Empty]"

        lines: list[str] = []
        for entry in entries:
            name = entry.get("name", entry.get("uri", "?"))
            kind = entry.get("type", "")
            line = f"  {'[dir] ' if kind == 'directory' else ''}{name}"
            lines.append(line)
        return f"Contents of {uri}:\n" + "\n".join(lines)

    return explore_memory


def make_add_knowledge_handler(state: MemoryState) -> Any:
    """Create handler for the ``add_knowledge`` tool."""

    async def add_knowledge(
        path: str,
        reason: str = "",
    ) -> str:
        if not state.client.available:
            return "[Memory unavailable]"

        uri = await state.client.add_resource(path=path, reason=reason or None)
        if uri is None:
            return f"[Failed to index {path}]"
        return f"Indexed: {path} → {uri}"

    return add_knowledge


def build_memory_tools(state: MemoryState) -> list[ToolInfo]:
    """Build the 4 memory tool definitions."""
    return [
        ToolInfo(
            name="recall_memory",
            description=(
                "Search your memories for relevant information. Use this to recall "
                "user preferences, past decisions, entity details, or prior context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["user", "agent"],
                        "description": "Search user memories or agent memories (default: user)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                    },
                },
                "required": ["query"],
            },
            handler=make_recall_handler(state),
            extension_name="memory",
        ),
        ToolInfo(
            name="save_memory",
            description=(
                "Save important information to persistent memory. Use this for user "
                "preferences, key decisions, entity details, or patterns worth remembering."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to save",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["preferences", "entities", "events", "cases", "patterns"],
                        "description": "Category of memory (default: preferences)",
                    },
                },
                "required": ["content"],
            },
            handler=make_save_handler(state),
            extension_name="memory",
        ),
        ToolInfo(
            name="explore_memory",
            description=(
                "Browse the memory filesystem to discover what memories exist. "
                "Useful for understanding what has been remembered."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "Memory URI to browse (default: viking://user/memories/)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List contents recursively (default: false)",
                    },
                },
            },
            handler=make_explore_handler(state),
            extension_name="memory",
        ),
        ToolInfo(
            name="add_knowledge",
            description=(
                "Index a local file or directory into the knowledge base so it "
                "can be recalled later."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory to index",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this resource is being indexed (optional)",
                    },
                },
                "required": ["path"],
            },
            handler=make_add_knowledge_handler(state),
            extension_name="memory",
        ),
    ]
