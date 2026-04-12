"""Tests for HarnessState and property delegation (Sprint 4)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from skillengine.agent import AgentConfig, AgentMessage, AgentRunner
from skillengine.harness_state import HarnessState
from skillengine.model_registry import TokenUsage
from skillengine.session.manager import SessionManager
from skillengine.session.models import CustomEntry


# ---------------------------------------------------------------------------
# HarnessState serialization
# ---------------------------------------------------------------------------


class TestHarnessStateSerialization:
    def test_to_session_entry(self):
        state = HarnessState(
            active_model="claude-sonnet",
            active_adapter_name="anthropic",
            thinking_level="high",
            cumulative_usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=10,
                cache_write_tokens=5,
            ),
        )
        data = state.to_session_entry()
        assert data["active_model"] == "claude-sonnet"
        assert data["active_adapter_name"] == "anthropic"
        assert data["thinking_level"] == "high"
        assert data["cumulative_usage"]["input_tokens"] == 100
        assert data["cumulative_usage"]["output_tokens"] == 50
        assert data["cumulative_usage"]["cache_read_tokens"] == 10
        assert data["cumulative_usage"]["cache_write_tokens"] == 5

    def test_from_session_data_roundtrip(self):
        original = HarnessState(
            active_model="gpt-4",
            active_adapter_name="openai",
            thinking_level="medium",
            cumulative_usage=TokenUsage(
                input_tokens=200, output_tokens=100,
            ),
        )
        data = original.to_session_entry()
        restored = HarnessState.from_session_data(data)
        assert restored.active_model == "gpt-4"
        assert restored.active_adapter_name == "openai"
        assert restored.thinking_level == "medium"
        assert restored.cumulative_usage.input_tokens == 200
        assert restored.cumulative_usage.output_tokens == 100

    def test_from_session_data_with_conversation(self):
        msg = MagicMock()
        data = {"active_model": "m"}
        state = HarnessState.from_session_data(data, conversation=[msg])
        assert len(state.conversation) == 1
        assert state.conversation[0] is msg

    def test_from_session_data_defaults(self):
        state = HarnessState.from_session_data({})
        assert state.active_model == ""
        assert state.active_adapter_name is None
        assert state.thinking_level == "off"
        assert state.cumulative_usage.input_tokens == 0


class TestHarnessStateDefaults:
    def test_transient_fields_created(self):
        state = HarnessState()
        assert isinstance(state.abort_event, asyncio.Event)
        assert isinstance(state.steering_queue, asyncio.Queue)
        assert isinstance(state.followup_queue, asyncio.Queue)
        assert state.conversation == []
        assert state.client is None


# ---------------------------------------------------------------------------
# Property delegation tests
# ---------------------------------------------------------------------------


class TestPropertyDelegation:
    def _make_runner(self):
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        return AgentRunner(engine, config)

    def test_conversation_delegates(self):
        runner = self._make_runner()
        assert runner._conversation is runner._state.conversation

    def test_conversation_setter(self):
        runner = self._make_runner()
        new_conv = [AgentMessage(role="user", content="hi")]
        runner._conversation = new_conv
        assert runner._state.conversation is new_conv

    def test_cumulative_usage_delegates(self):
        runner = self._make_runner()
        assert runner._cumulative_usage is runner._state.cumulative_usage

    def test_active_adapter_name_delegates(self):
        runner = self._make_runner()
        runner._active_adapter_name = "test"
        assert runner._state.active_adapter_name == "test"

    def test_abort_event_delegates(self):
        runner = self._make_runner()
        assert runner._abort_event is runner._state.abort_event

    def test_steering_queue_delegates(self):
        runner = self._make_runner()
        assert runner._steering_queue is runner._state.steering_queue

    def test_followup_queue_delegates(self):
        runner = self._make_runner()
        assert runner._followup_queue is runner._state.followup_queue

    def test_context_files_delegates(self):
        runner = self._make_runner()
        runner._context_files = ["file1"]
        assert runner._state.context_files == ["file1"]

    def test_client_delegates(self):
        runner = self._make_runner()
        mock_client = MagicMock()
        runner._client = mock_client
        assert runner._state.client is mock_client

    def test_on_skill_change_delegates(self):
        runner = self._make_runner()
        cb = MagicMock()
        runner._on_skill_change.append(cb)
        assert cb in runner._state.on_skill_change


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def _make_runner_with_session(self, tmp_path):
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        runner = AgentRunner(engine, config)

        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        mgr = SessionManager(session_dir=str(session_dir))
        runner._session_manager = mgr
        return runner, mgr

    def test_checkpoint_writes_to_session(self, tmp_path):
        runner, mgr = self._make_runner_with_session(tmp_path)
        runner._state.active_model = "test-model"
        runner._state.cumulative_usage = TokenUsage(
            input_tokens=50, output_tokens=25
        )
        runner.checkpoint()

        # Find the custom entry
        custom_entries = [
            e for e in mgr.entries
            if isinstance(e, CustomEntry) and e.custom_type == "harness_state"
        ]
        assert len(custom_entries) == 1
        data = custom_entries[0].data
        assert data["active_model"] == "test-model"
        assert data["cumulative_usage"]["input_tokens"] == 50

    def test_checkpoint_no_session_noop(self):
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(load_context_files=False)
        runner = AgentRunner(engine, config)
        # No session manager, should not raise
        runner.checkpoint()

    def test_state_init_from_config(self):
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        config = AgentConfig(
            model="custom-model",
            thinking_level="high",
            load_context_files=False,
        )
        runner = AgentRunner(engine, config)
        assert runner._state.active_model == "custom-model"
        assert runner._state.thinking_level == "high"
