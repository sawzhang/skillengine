"""SQLite-backed session and settings storage for the web UI."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SessionStorage:
    """SQLite-backed storage for sessions, settings, and API keys."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_dir = Path.home() / ".skillengine" / "web"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "storage.db"
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    created_at REAL,
                    updated_at REAL,
                    data TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provider_keys (
                    provider TEXT PRIMARY KEY,
                    api_key TEXT DEFAULT ''
                )
            """)
            conn.commit()

    def save_session(self, session_id: str, data: dict[str, Any], title: str = "") -> None:
        """Save or update a session."""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO sessions (id, title, created_at, updated_at, data)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   title=excluded.title, updated_at=excluded.updated_at, data=excluded.data""",
                (session_id, title, now, now, json.dumps(data)),
            )
            conn.commit()

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a session by ID."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, data FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "data": json.loads(row[4]),
            }

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions (metadata only)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_setting(self, key: str, default: str = "") -> str:
        """Get a setting value."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    def get_provider_key(self, provider: str) -> str | None:
        """Get an API key for a provider."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT api_key FROM provider_keys WHERE provider = ?", (provider,)
            ).fetchone()
            return row[0] if row else None

    def set_provider_key(self, provider: str, api_key: str) -> None:
        """Set an API key for a provider."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO provider_keys (provider, api_key) VALUES (?, ?)"
                " ON CONFLICT(provider)"
                " DO UPDATE SET api_key=excluded.api_key",
                (provider, api_key),
            )
            conn.commit()
