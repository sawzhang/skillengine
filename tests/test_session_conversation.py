"""Tests for session-conversation integration (Sprint 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
from skillengine.session.conversation import (
    agent_message_to_session_kwargs,
    session_entry_to_agent_message,
)
from skillengine.session.manager import SessionManager
from skillengine.session.models import SessionMessageEntry


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------


class TestSessionEntryToAgentMessage:
    def test_basic_conversion(self):
        entry = SessionMessageEntry(
            role="user",
            content="Hello world",
        )
        msg = session_entry_to_agent_message(entry)
        assert isinstance(msg, AgentMessage)
        assert msg.role == "user"
        assert msg.content == "Hello world"

    def test_tool_message_conversion(self):
        entry = SessionMessageEntry(
            role="tool",
            content="tool result",
            tool_call_id="tc_123",
            name="execute",
        )
        msg = session_entry_to_agent_message(entry)
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_123"
        assert msg.name == "execute"

    def test_assistant_with_tool_calls(self):
        entry = SessionMessageEntry(
            role="assistant",
            content="Let me help",
            tool_calls=[{"name": "read", "arguments": '{"path": "/tmp"}', "id": "tc_1"}],
        )
        msg = session_entry_to_agent_message(entry)
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read"

    def test_empty_tool_calls(self):
        entry = SessionMessageEntry(role="assistant", content="done")
        msg = session_entry_to_agent_message(entry)
        assert msg.tool_calls == []

    def test_metadata_preserved(self):
        entry = SessionMessageEntry(
            role="user",
            content="test",
            metadata={"source": "telegram"},
        )
        msg = session_entry_to_agent_message(entry)
        assert msg.metadata == {"source": "telegram"}


class TestAgentMessageToSessionKwargs:
    def test_basic_conversion(self):
        msg = AgentMessage(role="user", content="Hello")
        kwargs = agent_message_to_session_kwargs(msg)
        assert kwargs["role"] == "user"
        assert kwargs["content"] == "Hello"

    def test_tool_message(self):
        msg = AgentMessage(
            role="tool",
            content="result",
            tool_call_id="tc_1",
            name="execute",
        )
        kwargs = agent_message_to_session_kwargs(msg)
        assert kwargs["tool_call_id"] == "tc_1"
        assert kwargs["name"] == "execute"

    def test_assistant_with_tool_calls(self):
        msg = AgentMessage(
            role="assistant",
            content="I'll help",
            tool_calls=[{"name": "write", "arguments": "{}", "id": "tc_2"}],
        )
        kwargs = agent_message_to_session_kwargs(msg)
        assert kwargs["tool_calls"] == msg.tool_calls

    def test_no_optional_fields_when_empty(self):
        msg = AgentMessage(role="user", content="hi")
        kwargs = agent_message_to_session_kwargs(msg)
        assert "tool_calls" not in kwargs
        assert "tool_call_id" not in kwargs
        assert "name" not in kwargs
        assert "metadata" not in kwargs

    def test_multimodal_content_extracted(self):
        """Multi-modal content is serialized as text."""
        text_block = MagicMock()
        text_block.text = "Hello "
        text_block2 = MagicMock()
        text_block2.text = "World"
        msg = AgentMessage(role="user", content=[text_block, text_block2])
        kwargs = agent_message_to_session_kwargs(msg)
        assert kwargs["content"] == "Hello World"


class TestRoundtrip:
    def test_message_roundtrip(self):
        """AgentMessage -> session kwargs -> SessionMessageEntry -> AgentMessage."""
        original = AgentMessage(
            role="assistant",
            content="test content",
            tool_calls=[{"name": "read", "arguments": '{"path": "/x"}', "id": "tc_1"}],
            metadata={"key": "value"},
        )
        kwargs = agent_message_to_session_kwargs(original)
        entry = SessionMessageEntry(**kwargs)
        restored = session_entry_to_agent_message(entry)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_calls == original.tool_calls
        assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# _append_to_conversation tests
# ---------------------------------------------------------------------------


class TestAppendToConversation:
    def _make_runner(self):
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        return AgentRunner(engine, config)

    def test_appends_to_conversation(self):
        runner = self._make_runner()
        msg = AgentMessage(role="user", content="hello")
        runner._append_to_conversation(msg)
        assert len(runner._conversation) == 1
        assert runner._conversation[0] is msg

    def test_no_session_no_error(self):
        runner = self._make_runner()
        msg = AgentMessage(role="user", content="hello")
        runner._append_to_conversation(msg)
        # No session manager, should just append to conversation silently

    def test_dual_write_to_session(self):
        runner = self._make_runner()
        session_mgr = MagicMock()
        runner._session_manager = session_mgr
        msg = AgentMessage(role="user", content="hello")
        runner._append_to_conversation(msg)

        assert len(runner._conversation) == 1
        session_mgr.append_message.assert_called_once()
        call_kwargs = session_mgr.append_message.call_args[1]
        assert call_kwargs["role"] == "user"
        assert call_kwargs["content"] == "hello"

    def test_session_error_does_not_break_conversation(self):
        runner = self._make_runner()
        session_mgr = MagicMock()
        session_mgr.append_message.side_effect = RuntimeError("disk full")
        runner._session_manager = session_mgr

        msg = AgentMessage(role="user", content="hello")
        runner._append_to_conversation(msg)
        # Should still be in conversation despite session error
        assert len(runner._conversation) == 1


# ---------------------------------------------------------------------------
# wake() tests
# ---------------------------------------------------------------------------


class TestWake:
    def test_wake_restores_conversation(self, tmp_path):
        # Create a session with some messages
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        mgr = SessionManager(session_dir=str(session_dir))
        session_id = mgr.header.id
        mgr.append_message(role="user", content="Hello")
        mgr.append_message(role="assistant", content="Hi there!")

        # Wake from it
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        runner = AgentRunner.wake(
            session_id=session_id,
            session_dir=str(session_dir),
            engine=engine,
            config=config,
        )

        assert len(runner._conversation) == 2
        assert runner._conversation[0].role == "user"
        assert runner._conversation[0].content == "Hello"
        assert runner._conversation[1].role == "assistant"
        assert runner._conversation[1].content == "Hi there!"

    def test_wake_restores_model(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        mgr = SessionManager(session_dir=str(session_dir))
        session_id = mgr.header.id
        mgr.append_message(role="user", content="hello")
        mgr.append_model_change(
            prev_model="old-model",
            new_model="new-model",
        )

        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        runner = AgentRunner.wake(
            session_id=session_id,
            session_dir=str(session_dir),
            engine=engine,
            config=config,
        )
        assert runner.config.model == "new-model"

    def test_wake_session_not_found(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None

        with pytest.raises(FileNotFoundError):
            AgentRunner.wake(
                session_id="nonexistent-id",
                session_dir=str(session_dir),
                engine=engine,
            )

    def test_wake_sets_session_manager(self, tmp_path):
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        mgr = SessionManager(session_dir=str(session_dir))
        session_id = mgr.header.id

        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        runner = AgentRunner.wake(
            session_id=session_id,
            session_dir=str(session_dir),
            engine=engine,
            config=config,
        )
        assert runner._session_manager is not None
        assert runner._session_manager.header.id == session_id
