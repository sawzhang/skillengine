"""Tests for AgentDefinition, RuntimeConfig, and Environment (Sprint 2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from skillengine.agent import AgentConfig, AgentRunner
from skillengine.definition import AgentDefinition, RuntimeConfig
from skillengine.environment import Environment


class TestAgentDefinition:
    def test_frozen(self):
        d = AgentDefinition(name="test", model="gpt-4")
        with pytest.raises(AttributeError):
            d.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        d = AgentDefinition()
        assert d.name == "default"
        assert d.model == "MiniMax-M2.1"
        assert d.enable_tools is True
        assert d.skill_dirs == ()
        assert d.skill_description_budget == 16000

    def test_skill_dirs_tuple(self):
        d = AgentDefinition(skill_dirs=(Path("/a"), Path("/b")))
        assert isinstance(d.skill_dirs, tuple)
        assert len(d.skill_dirs) == 2


class TestRuntimeConfig:
    def test_mutable(self):
        r = RuntimeConfig()
        r.max_turns = 50
        assert r.max_turns == 50

    def test_defaults(self):
        r = RuntimeConfig()
        assert r.max_turns == 20
        assert r.temperature == 1.0
        assert r.transport == "sse"
        assert r.cache_retention == "short"
        assert r.thinking_level is None


class TestAgentConfigConversion:
    def test_to_definition(self):
        config = AgentConfig(
            model="claude-sonnet",
            system_prompt="You are helpful",
            skill_dirs=[Path("/skills")],
            enable_tools=False,
            skill_description_budget=8000,
        )
        defn = config.to_definition()
        assert isinstance(defn, AgentDefinition)
        assert defn.model == "claude-sonnet"
        assert defn.system_prompt == "You are helpful"
        assert defn.skill_dirs == (Path("/skills"),)
        assert defn.enable_tools is False
        assert defn.skill_description_budget == 8000

    def test_to_runtime_config(self):
        config = AgentConfig(
            base_url="http://localhost:8080",
            api_key="sk-test",
            max_turns=10,
            temperature=0.5,
            cache_retention="long",
            session_id="sess-1",
        )
        rt = config.to_runtime_config()
        assert isinstance(rt, RuntimeConfig)
        assert rt.base_url == "http://localhost:8080"
        assert rt.api_key == "sk-test"
        assert rt.max_turns == 10
        assert rt.temperature == 0.5
        assert rt.cache_retention == "long"
        assert rt.session_id == "sess-1"

    def test_from_parts_roundtrip(self):
        original = AgentConfig(
            model="gpt-4",
            system_prompt="test",
            skill_dirs=[Path("/a")],
            enable_tools=True,
            enable_reasoning=False,
            base_url="http://x",
            api_key="key",
            max_turns=5,
            temperature=0.7,
            max_tokens=2048,
            transport="sse",
            cache_retention="none",
            session_id="s1",
            watch_skills=True,
            load_context_files=False,
            skill_description_budget=4000,
        )
        defn = original.to_definition()
        rt = original.to_runtime_config()
        rebuilt = AgentConfig.from_parts(defn, rt)

        assert rebuilt.model == original.model
        assert rebuilt.system_prompt == original.system_prompt
        assert rebuilt.skill_dirs == original.skill_dirs
        assert rebuilt.enable_tools == original.enable_tools
        assert rebuilt.enable_reasoning == original.enable_reasoning
        assert rebuilt.base_url == original.base_url
        assert rebuilt.api_key == original.api_key
        assert rebuilt.max_turns == original.max_turns
        assert rebuilt.temperature == original.temperature
        assert rebuilt.max_tokens == original.max_tokens
        assert rebuilt.transport == original.transport
        assert rebuilt.cache_retention == original.cache_retention
        assert rebuilt.session_id == original.session_id
        assert rebuilt.watch_skills == original.watch_skills
        assert rebuilt.load_context_files == original.load_context_files
        assert rebuilt.skill_description_budget == original.skill_description_budget

    def test_from_parts_default_runtime(self):
        defn = AgentDefinition(model="test-model")
        config = AgentConfig.from_parts(defn)
        assert config.model == "test-model"
        assert config.max_turns == 20  # RuntimeConfig default


class TestAgentConfigFromEnv:
    @pytest.fixture
    def clean_env(self, monkeypatch):
        for var in (
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "MINIMAX_MODEL",
            "ASE_CACHE_RETENTION",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_pure_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("MINIMAX_MODEL", "env-model")
        monkeypatch.setenv("ASE_CACHE_RETENTION", "long")

        config = AgentConfig.from_env()
        assert config.base_url == "https://env.example.com/v1"
        assert config.api_key == "sk-env"
        assert config.model == "env-model"
        assert config.cache_retention == "long"

    def test_overrides_win_over_env(self, clean_env, monkeypatch):
        """Regression: explicit overrides must not collide with env-derived kwargs."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("MINIMAX_MODEL", "env-model")

        config = AgentConfig.from_env(
            base_url="https://override.example.com/v1",
            api_key="sk-override",
            model="override-model",
            cache_retention="none",
        )
        assert config.base_url == "https://override.example.com/v1"
        assert config.api_key == "sk-override"
        assert config.model == "override-model"
        assert config.cache_retention == "none"

    def test_partial_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("MINIMAX_MODEL", "env-model")

        config = AgentConfig.from_env(model="override-model")
        assert config.base_url == "https://env.example.com/v1"
        assert config.api_key == "sk-env"
        assert config.model == "override-model"

    def test_invalid_cache_retention_falls_back(self, clean_env, monkeypatch):
        monkeypatch.setenv("ASE_CACHE_RETENTION", "garbage")
        config = AgentConfig.from_env()
        assert config.cache_retention == "short"

    def test_non_env_overrides_passthrough(self, clean_env):
        config = AgentConfig.from_env(
            max_turns=42,
            temperature=0.3,
            enable_tools=False,
        )
        assert config.max_turns == 42
        assert config.temperature == 0.3
        assert config.enable_tools is False


class TestEnvironment:
    def test_creation(self):
        engine = MagicMock()
        env = Environment(engine=engine)
        assert env.engine is engine
        assert env.adapter_registry is not None
        assert env.event_bus is not None
        assert env.model_registry is None
        assert env.context_manager is None


class TestAgentRunnerFromDefinition:
    def test_from_definition_creates_runner(self):
        defn = AgentDefinition(
            model="test-model",
            system_prompt="hello",
            load_context_files=False,
        )
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        env = Environment(engine=engine)

        runner = AgentRunner.from_definition(defn, env)
        assert isinstance(runner, AgentRunner)
        assert runner.config.model == "test-model"
        assert runner.config.system_prompt == "hello"
        assert runner.engine is engine

    def test_from_definition_with_runtime_config(self):
        defn = AgentDefinition(model="m", load_context_files=False)
        rt = RuntimeConfig(max_turns=3, temperature=0.1)
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        env = Environment(engine=engine)

        runner = AgentRunner.from_definition(defn, env, runtime_config=rt)
        assert runner.config.max_turns == 3
        assert runner.config.temperature == 0.1

    def test_from_definition_inherits_environment(self):
        defn = AgentDefinition(load_context_files=False)
        engine = MagicMock()
        engine.config = MagicMock()
        engine.extensions = None
        env = Environment(engine=engine)

        runner = AgentRunner.from_definition(defn, env)
        assert runner.adapter_registry is env.adapter_registry
        assert runner.events is env.event_bus
