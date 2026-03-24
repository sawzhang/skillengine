"""Web UI server using Starlette."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
    from starlette.websockets import WebSocket
except ImportError:
    raise ImportError("Web UI requires the 'web' extra. Install with: pip install skillengine[web]")

from skillengine.web.storage import SessionStorage

STATIC_DIR = Path(__file__).parent / "static"


def create_app(agent: Any = None, storage: SessionStorage | None = None) -> Starlette:
    """Create the web UI Starlette application.

    Args:
        agent: AgentRunner instance to use for chat
        storage: Session storage backend (defaults to SQLite)
    """
    _storage = storage or SessionStorage()
    _agent = agent

    async def index(request: Request) -> HTMLResponse:
        """Serve the main page."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text())
        return HTMLResponse("<h1>SkillEngine</h1><p>Static files not found.</p>")

    async def api_chat_stream(request: Request) -> Any:
        """SSE endpoint for streaming chat responses."""
        from starlette.responses import StreamingResponse

        body = await request.json()
        message = body.get("message", "")

        if not _agent or not message:
            return JSONResponse({"error": "No agent or message"}, status_code=400)

        async def event_generator():
            async for event in _agent.chat_stream_events(message):
                event_dict = {
                    "type": event.type,
                    "content": event.content,
                }
                if event.tool_name:
                    event_dict["tool_name"] = event.tool_name
                if event.tool_call_id:
                    event_dict["tool_call_id"] = event.tool_call_id
                if event.error:
                    event_dict["error"] = event.error
                if event.finish_reason:
                    event_dict["finish_reason"] = event.finish_reason
                if event.args_delta:
                    event_dict["args_delta"] = event.args_delta
                yield f"data: {json.dumps(event_dict)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def api_sessions(request: Request) -> JSONResponse:
        """List or manage sessions."""
        if request.method == "GET":
            sessions = _storage.list_sessions()
            return JSONResponse({"sessions": sessions})
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    async def api_session_detail(request: Request) -> JSONResponse:
        """Get session details."""
        session_id = request.path_params["session_id"]
        session = _storage.load_session(session_id)
        if session is None:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse(session)

    async def api_skills(request: Request) -> JSONResponse:
        """List available skills."""
        if not _agent:
            return JSONResponse({"skills": []})
        skills_list = [
            {
                "name": s.name,
                "description": s.description,
                "emoji": s.metadata.emoji,
            }
            for s in _agent.skills
        ]
        return JSONResponse({"skills": skills_list})

    async def api_config(request: Request) -> JSONResponse:
        """Get agent configuration."""
        if not _agent:
            return JSONResponse({"error": "No agent"}, status_code=400)
        return JSONResponse(
            {
                "model": _agent.config.model,
                "thinking_level": _agent.config.thinking_level or "off",
                "max_turns": _agent.config.max_turns,
            }
        )

    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for bidirectional communication."""
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                cmd_type = data.get("type", "")

                if cmd_type == "prompt":
                    message = data.get("message", "")
                    if _agent and message:
                        async for event in _agent.chat_stream_events(message):
                            event_dict = {"type": event.type, "content": event.content}
                            if event.tool_name:
                                event_dict["tool_name"] = event.tool_name
                            if event.finish_reason:
                                event_dict["finish_reason"] = event.finish_reason
                            await websocket.send_json(event_dict)

                elif cmd_type == "abort":
                    if _agent:
                        _agent.abort()
                        await websocket.send_json({"type": "aborted"})

                elif cmd_type == "clear":
                    if _agent:
                        _agent.clear_history()
                        _agent.reset_abort()
                        await websocket.send_json({"type": "cleared"})

                else:
                    await websocket.send_json({"type": "error", "error": f"Unknown: {cmd_type}"})

        except Exception:
            pass

    routes = [
        Route("/", index),
        Route("/api/chat/stream", api_chat_stream, methods=["POST"]),
        Route("/api/sessions", api_sessions, methods=["GET"]),
        Route("/api/sessions/{session_id}", api_session_detail, methods=["GET"]),
        Route("/api/skills", api_skills, methods=["GET"]),
        Route("/api/config", api_config, methods=["GET"]),
    ]

    # Only mount static files if the directory exists
    if STATIC_DIR.exists():
        routes.append(Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"))

    app = Starlette(
        routes=routes,
        on_startup=[],
    )

    # Register WebSocket separately
    app.add_websocket_route("/ws", websocket_endpoint)

    return app


def run_server(agent: Any = None, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the web UI server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Running the web server requires uvicorn. Install with: pip install skillengine[web]"
        )

    app = create_app(agent=agent)
    uvicorn.run(app, host=host, port=port)
