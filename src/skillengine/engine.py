"""
Main Skills Engine implementation.

Provides a high-level API for loading, filtering, and managing skills.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from watchfiles import awatch

from skillengine.config import SkillsConfig
from skillengine.extensions.manager import ExtensionManager
from skillengine.filters import DefaultSkillFilter, SkillFilter
from skillengine.filters.base import FilterContext
from skillengine.loaders import MarkdownSkillLoader, SkillLoader
from skillengine.logging import get_logger
from skillengine.models import Skill, SkillSnapshot, SkillSource
from skillengine.runtime import BashRuntime, ExecutionResult, SkillRuntime

logger = get_logger("engine")

# Thread-local storage for environment variable backups
_env_backup_var: contextvars.ContextVar[dict[str, str | None]] = contextvars.ContextVar(
    "env_backup", default={}
)

# Lock for protecting os.environ modifications
_env_lock = threading.Lock()


class SkillsEngine:
    """
    Main skills engine providing a unified API for skill management.

    Example:
        engine = SkillsEngine(
            config=SkillsConfig(
                skill_dirs=[Path("./skills")],
                watch=True,
            )
        )

        # Load and filter skills
        snapshot = engine.get_snapshot()

        # Get prompt for LLM
        print(snapshot.prompt)

        # Execute a skill command
        result = await engine.execute("echo 'hello'")
    """

    def __init__(
        self,
        config: SkillsConfig | None = None,
        loader: SkillLoader | None = None,
        filter: SkillFilter | None = None,
        runtime: SkillRuntime | None = None,
    ) -> None:
        self.config = config or SkillsConfig()
        self.loader = loader or MarkdownSkillLoader()
        self.filter = filter or DefaultSkillFilter()
        self.runtime = runtime or BashRuntime()

        self._snapshot: SkillSnapshot | None = None
        self._snapshot_version = 0
        self.extensions: ExtensionManager | None = None

        # File watching state
        self._watch_task: asyncio.Task[None] | None = None
        self._watch_stop_event: asyncio.Event | None = None
        self._watch_callbacks: list[Callable[[set[Path]], None]] = []

    def load_skills(self) -> list[Skill]:
        """
        Load all skills from configured directories.

        Returns:
            List of loaded skills (unfiltered)
        """
        skills: list[Skill] = []
        seen_names: set[str] = set()

        # Load in priority order (later overrides earlier)
        dirs = self.config.merge_dirs()
        logger.debug("Loading skills from %d directories", len(dirs))

        for directory in dirs:
            source = self._resolve_source(directory)
            logger.debug("Scanning directory: %s (source=%s)", directory, source.value)
            entries = self.loader.load_directory(directory, source)

            for entry in entries:
                if entry.skill and not entry.load_error:
                    # Remove previous if exists (for override)
                    if entry.skill.name in seen_names:
                        logger.debug("Overriding skill: %s", entry.skill.name)
                        skills = [s for s in skills if s.name != entry.skill.name]
                    skills.append(entry.skill)
                    seen_names.add(entry.skill.name)
                elif entry.load_error:
                    logger.warning("Failed to load skill: %s", entry.load_error)

        logger.info("Loaded %d skills", len(skills))
        return skills

    def filter_skills(
        self,
        skills: list[Skill] | None = None,
        context: FilterContext | None = None,
    ) -> list[Skill]:
        """
        Filter skills based on eligibility.

        Args:
            skills: Skills to filter (loads if not provided)
            context: Filter context (auto-detected if not provided)

        Returns:
            List of eligible skills
        """
        if skills is None:
            skills = self.load_skills()

        if context is None:
            context = self._build_context()

        return self.filter.get_eligible(skills, self.config, context)

    def get_snapshot(self, force_reload: bool = False) -> SkillSnapshot:
        """
        Get a snapshot of eligible skills.

        Caches the snapshot for efficiency. Use force_reload=True to refresh.

        Args:
            force_reload: Force reloading skills

        Returns:
            SkillSnapshot with eligible skills and formatted prompt
        """
        if self._snapshot is not None and not force_reload:
            return self._snapshot

        skills = self.load_skills()
        eligible = self.filter_skills(skills)

        # Filter out skills that shouldn't appear in prompt
        visible = [s for s in eligible if not s.metadata.invocation.disable_model_invocation]

        self._snapshot_version += 1
        self._snapshot = SkillSnapshot(
            skills=eligible,
            prompt=self.format_prompt(visible),
            version=self._snapshot_version,
            source_dirs=self.config.merge_dirs(),
        )

        logger.debug(
            "Created snapshot v%d with %d eligible skills (%d visible)",
            self._snapshot_version,
            len(eligible),
            len(visible),
        )
        return self._snapshot

    def format_prompt(
        self,
        skills: list[Skill] | None = None,
        format: str | None = None,
    ) -> str:
        """
        Format skills as a prompt for LLM.

        Args:
            skills: Skills to format (uses snapshot if not provided)
            format: Format type ("xml", "markdown", "json")

        Returns:
            Formatted prompt string
        """
        if skills is None:
            snapshot = self.get_snapshot()
            skills = [
                s for s in snapshot.skills if not s.metadata.invocation.disable_model_invocation
            ]

        format = format or self.config.prompt_format

        if format == "xml":
            return self._format_xml(skills)
        elif format == "markdown":
            return self._format_markdown(skills)
        elif format == "json":
            return self._format_json(skills)
        else:
            return self._format_xml(skills)

    def _format_xml(self, skills: list[Skill]) -> str:
        """Format skills as XML (SkillEngine standard)."""
        if not skills:
            return ""

        lines = ["<skills>"]
        for skill in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{xml_escape(skill.name)}</name>")
            lines.append(f"    <description>{xml_escape(skill.description)}</description>")
            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _format_markdown(self, skills: list[Skill]) -> str:
        """Format skills as Markdown."""
        if not skills:
            return ""

        lines = ["## Available Skills", ""]
        for skill in skills:
            emoji = skill.metadata.emoji or "🔧"
            lines.append(f"- **{emoji} {skill.name}**: {skill.description}")

        return "\n".join(lines)

    def _format_json(self, skills: list[Skill]) -> str:
        """Format skills as JSON."""
        import json

        data = [{"name": s.name, "description": s.description} for s in skills]
        return json.dumps(data, indent=2)

    @contextmanager
    def env_context(self, skills: list[Skill] | None = None) -> Iterator[None]:
        """
        Context manager to apply skill environment overrides.

        Temporarily sets environment variables based on skill configs,
        and restores them on exit.

        This method is thread-safe: each thread/context gets its own backup
        of environment variables, and modifications to os.environ are protected
        by a lock.

        Example:
            with engine.env_context():
                # GITHUB_TOKEN is set from skill config
                result = await engine.execute("gh pr list")
        """
        if skills is None:
            snapshot = self.get_snapshot()
            skills = snapshot.skills

        # Create a fresh backup dict for this context
        backup: dict[str, str | None] = {}
        token = _env_backup_var.set(backup)

        try:
            # Apply overrides (thread-safe)
            self._apply_env_overrides(skills)
            yield
        finally:
            # Restore original values (thread-safe)
            self._restore_env()
            _env_backup_var.reset(token)

    def _apply_env_overrides(self, skills: list[Skill]) -> None:
        """Apply environment variable overrides from skill configs."""
        for skill in skills:
            skill_config = self.config.get_skill_config(skill.skill_key)

            # Apply explicit env overrides
            for key, value in skill_config.env.items():
                self._set_env(key, value)

            # Apply API key to primary env
            if skill.metadata.primary_env and skill_config.api_key:
                self._set_env(skill.metadata.primary_env, skill_config.api_key)

    def _set_env(self, key: str, value: str) -> None:
        """Set an environment variable, backing up the original (thread-safe)."""
        backup = _env_backup_var.get()
        with _env_lock:
            if key not in backup:
                backup[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_env(self) -> None:
        """Restore backed up environment variables (thread-safe)."""
        backup = _env_backup_var.get()
        with _env_lock:
            for key, original in backup.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original
            backup.clear()

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: Callable[[str], None] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute a command using the runtime.

        Args:
            command: Command to execute
            cwd: Working directory
            env: Additional environment variables
            timeout: Execution timeout
            on_output: Callback for streaming output lines
            abort_signal: Event to signal abort

        Returns:
            ExecutionResult with output or error
        """
        logger.debug("Executing command: %s", command[:100])
        result = await self.runtime.execute(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout or self.config.default_timeout_seconds,
            on_output=on_output,
            abort_signal=abort_signal,
        )
        if result.success:
            logger.debug("Command succeeded (exit_code=%d)", result.exit_code)
        else:
            logger.warning("Command failed: %s", result.error)
        return result

    async def execute_script(
        self,
        script: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: Callable[[str], None] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """
        Execute a script using the runtime.

        Args:
            script: Script content
            cwd: Working directory
            env: Additional environment variables
            timeout: Execution timeout
            on_output: Callback for streaming output lines
            abort_signal: Event to signal abort

        Returns:
            ExecutionResult with output or error
        """
        return await self.runtime.execute_script(
            script=script,
            cwd=cwd,
            env=env,
            timeout=timeout or self.config.default_timeout_seconds,
            on_output=on_output,
            abort_signal=abort_signal,
        )

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name from the current snapshot."""
        snapshot = self.get_snapshot()
        return snapshot.get_skill(name)

    async def execute_action(
        self,
        skill_name: str,
        action_name: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute a skill action directly, without LLM.

        This is the deterministic execution path: it resolves the action's
        script and runs it with the given arguments.

        Args:
            skill_name: Name of the skill
            action_name: Name of the action within the skill
            args: CLI arguments to pass to the script
            env: Additional environment variables
            timeout: Execution timeout

        Returns:
            ExecutionResult with output or error
        """
        skill = self.get_skill(skill_name)
        if skill is None:
            return ExecutionResult(
                exit_code=1,
                output="",
                error=f"Skill not found: {skill_name}",
                success=False,
            )

        action = skill.get_action(action_name)
        if action is None:
            available = ", ".join(skill.actions.keys()) if skill.actions else "none"
            return ExecutionResult(
                exit_code=1,
                output="",
                error=(
                    f"Action '{action_name}' not found in skill "
                    f"'{skill_name}'. Available: {available}"
                ),
                success=False,
            )

        # Resolve script path relative to skill's base_dir
        script_path = skill.base_dir / action.script
        if not script_path.exists():
            return ExecutionResult(
                exit_code=1,
                output="",
                error=f"Script not found: {script_path}",
                success=False,
            )

        # Build command
        cmd_parts = ["python", str(script_path)]
        if args:
            cmd_parts.extend(args)
        command = " ".join(cmd_parts)

        logger.debug("Executing action %s.%s: %s", skill_name, action_name, command)

        return await self.execute(
            command=command,
            cwd=str(skill.base_dir),
            env=env,
            timeout=timeout,
        )

    def invalidate_cache(self) -> None:
        """Invalidate the cached snapshot."""
        self._snapshot = None

    def init_extensions(self) -> ExtensionManager:
        """
        Initialize the extension system, discover and load all extensions.

        Returns:
            The ExtensionManager instance
        """
        self.extensions = ExtensionManager(self)
        self.extensions.load_all()
        return self.extensions

    def _resolve_source(self, directory: Path) -> SkillSource:
        """Resolve the source type for a directory."""
        if self.config.bundled_dir and directory == self.config.bundled_dir:
            return SkillSource.BUNDLED
        if self.config.managed_dir and directory == self.config.managed_dir:
            return SkillSource.MANAGED
        return SkillSource.WORKSPACE

    def _build_context(self) -> FilterContext:
        """Build a filter context from the current environment."""
        return FilterContext(
            platform=sys.platform,
            env_vars=set(os.environ.keys()),
        )

    # File watching methods

    async def start_watching(
        self,
        callback: Callable[[set[Path]], None] | None = None,
    ) -> None:
        """
        Start watching skill directories for changes.

        When SKILL.md files change, the cache is invalidated and callbacks are invoked.

        Args:
            callback: Optional callback to invoke with changed paths

        Example:
            async def on_change(changed: Set[Path]):
                print(f"Skills changed: {changed}")

            await engine.start_watching(on_change)
        """
        if self._watch_task is not None:
            return  # Already watching

        if callback:
            self._watch_callbacks.append(callback)

        self._watch_stop_event = asyncio.Event()
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watching(self) -> None:
        """Stop watching skill directories."""
        if self._watch_task is None:
            return

        if self._watch_stop_event:
            self._watch_stop_event.set()

        self._watch_task.cancel()
        try:
            await self._watch_task
        except asyncio.CancelledError:
            pass

        self._watch_task = None
        self._watch_stop_event = None

    def add_watch_callback(self, callback: Callable[[set[Path]], None]) -> None:
        """Add a callback to be invoked when skills change."""
        self._watch_callbacks.append(callback)

    def remove_watch_callback(self, callback: Callable[[set[Path]], None]) -> None:
        """Remove a watch callback."""
        if callback in self._watch_callbacks:
            self._watch_callbacks.remove(callback)

    @property
    def is_watching(self) -> bool:
        """Check if file watching is active."""
        return self._watch_task is not None and not self._watch_task.done()

    async def _watch_loop(self) -> None:
        """Internal watch loop that monitors skill directories."""
        dirs = self.config.merge_dirs()
        if not dirs:
            return

        # Convert to strings for watchfiles
        watch_paths = [str(d) for d in dirs if d.exists()]
        if not watch_paths:
            return

        try:
            async for changes in awatch(
                *watch_paths,
                debounce=self.config.watch_debounce_ms,
                stop_event=self._watch_stop_event,
            ):
                # Filter to only SKILL.md changes
                skill_changes: set[Path] = set()
                for change_type, path_str in changes:
                    path = Path(path_str)
                    if path.name == "SKILL.md":
                        skill_changes.add(path)

                if skill_changes:
                    # Invalidate cache
                    self.invalidate_cache()

                    # Invoke callbacks
                    for callback in self._watch_callbacks:
                        try:
                            callback(skill_changes)
                        except Exception:
                            pass  # Don't let callback errors stop watching
        except asyncio.CancelledError:
            pass
