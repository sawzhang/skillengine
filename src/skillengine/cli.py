"""
Command-line interface for the skills engine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from skillengine.config import SkillsConfig
from skillengine.engine import SkillsEngine
from skillengine.loaders import MarkdownSkillLoader
from skillengine.logging import setup_logging
from skillengine.models import SkillSource

console = Console()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SkillEngine CLI",
        prog="skills",
    )

    # Global options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (debug logging)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available skills")
    list_parser.add_argument(
        "-d",
        "--dir",
        action="append",
        dest="dirs",
        help="Skill directories to scan",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all skills (including ineligible)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Show command
    show_parser = subparsers.add_parser("show", help="Show skill details")
    show_parser.add_argument("name", help="Skill name")
    show_parser.add_argument(
        "-d",
        "--dir",
        action="append",
        dest="dirs",
        help="Skill directories to scan",
    )

    # Prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate skills prompt")
    prompt_parser.add_argument(
        "-d",
        "--dir",
        action="append",
        dest="dirs",
        help="Skill directories to scan",
    )
    prompt_parser.add_argument(
        "-f",
        "--format",
        choices=["xml", "markdown", "json"],
        default="xml",
        help="Prompt format",
    )

    # Execute command
    exec_parser = subparsers.add_parser("exec", help="Execute a command")
    exec_parser.add_argument("command", nargs="+", help="Command to execute")
    exec_parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate SKILL.md files")
    validate_parser.add_argument(
        "-d",
        "--dir",
        action="append",
        dest="dirs",
        help="Skill directories to validate",
    )

    # Config command with subcommands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config commands")

    # config show
    config_subparsers.add_parser("show", help="Show current configuration")

    # config init
    config_init_parser = config_subparsers.add_parser("init", help="Initialize a new config file")
    config_init_parser.add_argument(
        "-o",
        "--output",
        default="skills-config.yaml",
        help="Output file path",
    )

    # config path
    config_subparsers.add_parser("path", help="Show config file paths")

    # Ext command with subcommands
    ext_parser = subparsers.add_parser("ext", help="Extension management")
    ext_subparsers = ext_parser.add_subparsers(dest="ext_command", help="Extension commands")

    ext_list_parser = ext_subparsers.add_parser("list", help="List installed extensions")
    ext_list_parser.add_argument(
        "-d", "--dir", action="append", dest="dirs", help="Skill directories"
    )

    ext_info_parser = ext_subparsers.add_parser("info", help="Show extension details")
    ext_info_parser.add_argument("name", help="Extension name")
    ext_info_parser.add_argument(
        "-d", "--dir", action="append", dest="dirs", help="Skill directories"
    )

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload skills and extensions")
    reload_parser.add_argument(
        "-d", "--dir", action="append", dest="dirs", help="Skill directories"
    )

    # Prompts command with subcommands
    prompts_parser = subparsers.add_parser("prompts", help="Prompt template management")
    prompts_subparsers = prompts_parser.add_subparsers(
        dest="prompts_command", help="Prompt commands"
    )

    prompts_subparsers.add_parser("list", help="List available prompt templates")

    prompts_show_parser = prompts_subparsers.add_parser("show", help="Show prompt template content")
    prompts_show_parser.add_argument("name", help="Template name")

    # Commands command
    commands_parser = subparsers.add_parser(
        "commands", help="List all slash commands from all sources"
    )
    commands_parser.add_argument(
        "-d", "--dir", action="append", dest="dirs", help="Skill directories"
    )

    # Chat command (interactive or mode-based)
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat or run in a mode")
    chat_parser.add_argument("-d", "--dir", action="append", dest="dirs", help="Skill directories")
    chat_parser.add_argument(
        "--mode",
        choices=["interactive", "json", "rpc"],
        default="interactive",
        help="Execution mode (default: interactive)",
    )
    chat_parser.add_argument("--model", default=None, help="Model to use")
    chat_parser.add_argument(
        "prompt_text", nargs="?", default=None, help="Prompt text (for json mode)"
    )

    # Serve command (web UI)
    serve_parser = subparsers.add_parser("serve", help="Start the web UI server")
    serve_parser.add_argument("-d", "--dir", action="append", dest="dirs", help="Skill directories")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind to")

    args = parser.parse_args()

    # Setup logging based on verbosity
    if getattr(args, "verbose", False):
        setup_logging("DEBUG")
    else:
        setup_logging("WARNING")

    if args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "prompt":
        cmd_prompt(args)
    elif args.command == "exec":
        asyncio.run(cmd_exec(args))
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "ext":
        cmd_ext(args)
    elif args.command == "reload":
        cmd_reload(args)
    elif args.command == "prompts":
        cmd_prompts(args)
    elif args.command == "commands":
        cmd_commands(args)
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


def _create_engine(dirs: list[str] | None = None) -> SkillsEngine:
    """Create a skills engine from CLI args."""
    skill_dirs = [Path(d) for d in (dirs or [])]
    if not skill_dirs:
        # Default to current directory's skills folder
        skill_dirs = [Path.cwd() / "skills"]

    config = SkillsConfig(skill_dirs=skill_dirs)
    return SkillsEngine(config=config)


def cmd_list(args: argparse.Namespace) -> None:
    """List available skills."""
    engine = _create_engine(args.dirs)

    if args.all:
        skills = engine.load_skills()
    else:
        skills = engine.filter_skills()

    if args.json:
        import json

        data = [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source.value,
                "emoji": s.metadata.emoji,
            }
            for s in skills
        ]
        console.print_json(json.dumps(data, indent=2))
        return

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for skill in skills:
        emoji = skill.metadata.emoji or "🔧"
        name = f"{emoji} {skill.name}"
        table.add_row(name, skill.description[:60], skill.source.value)

    console.print(table)
    console.print(f"\n[dim]Total: {len(skills)} skills[/dim]")


def cmd_show(args: argparse.Namespace) -> None:
    """Show skill details."""
    engine = _create_engine(args.dirs)
    skills = engine.load_skills()

    skill = next((s for s in skills if s.name == args.name), None)
    if not skill:
        console.print(f"[red]Skill not found: {args.name}[/red]")
        sys.exit(1)

    emoji = skill.metadata.emoji or "🔧"
    console.print(f"\n[bold]{emoji} {skill.name}[/bold]")
    console.print(f"[dim]{skill.description}[/dim]\n")

    # Metadata
    console.print("[bold]Metadata:[/bold]")
    console.print(f"  Source: {skill.source.value}")
    console.print(f"  File: {skill.file_path}")

    if skill.metadata.homepage:
        console.print(f"  Homepage: {skill.metadata.homepage}")

    if skill.metadata.requires.bins:
        console.print(f"  Required bins: {', '.join(skill.metadata.requires.bins)}")

    if skill.metadata.requires.env:
        console.print(f"  Required env: {', '.join(skill.metadata.requires.env)}")

    # Content
    console.print("\n[bold]Content:[/bold]")
    console.print(skill.content[:500])
    if len(skill.content) > 500:
        console.print("[dim]... (truncated)[/dim]")


def cmd_prompt(args: argparse.Namespace) -> None:
    """Generate skills prompt."""
    engine = _create_engine(args.dirs)
    engine.config.prompt_format = args.format

    snapshot = engine.get_snapshot()
    console.print(snapshot.prompt)


async def cmd_exec(args: argparse.Namespace) -> None:
    """Execute a command."""
    engine = _create_engine(None)
    command = " ".join(args.command)

    result = await engine.execute(command, timeout=args.timeout)

    if result.success:
        console.print(result.output)
    else:
        console.print(f"[red]Error:[/red] {result.error}")
        if result.output:
            console.print(result.output)
        sys.exit(result.exit_code)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate SKILL.md files."""
    dirs = [Path(d) for d in (args.dirs or [])]
    if not dirs:
        dirs = [Path.cwd() / "skills"]

    loader = MarkdownSkillLoader()
    errors: list[tuple[Path, str]] = []
    valid_count = 0

    for directory in dirs:
        if not directory.exists():
            console.print(f"[yellow]Directory not found: {directory}[/yellow]")
            continue

        console.print(f"\n[bold]Validating skills in {directory}[/bold]")

        entries = loader.load_directory(directory, SkillSource.WORKSPACE)

        for entry in entries:
            if entry.load_error:
                file_path = entry.skill.file_path if entry.skill else directory
                errors.append((file_path, entry.load_error))
                console.print(f"  [red]✗[/red] {entry.load_error}")
            elif entry.skill:
                valid_count += 1
                # Check for common issues
                warnings = []

                if not entry.skill.description:
                    warnings.append("Missing description")

                if not entry.skill.metadata.emoji:
                    warnings.append("No emoji set")

                if entry.skill.metadata.requires.bins:
                    for bin_name in entry.skill.metadata.requires.bins:
                        import shutil

                        if not shutil.which(bin_name):
                            warnings.append(f"Binary '{bin_name}' not found in PATH")

                if warnings:
                    console.print(f"  [yellow]⚠[/yellow] {entry.skill.name}: {', '.join(warnings)}")
                else:
                    console.print(f"  [green]✓[/green] {entry.skill.name}")

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Valid: {valid_count}")
    console.print(f"  Errors: {len(errors)}")

    if errors:
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """Configuration management commands."""
    if args.config_command == "show":
        _config_show()
    elif args.config_command == "init":
        _config_init(args.output)
    elif args.config_command == "path":
        _config_path()
    else:
        console.print("[yellow]Usage: skills config <show|init|path>[/yellow]")


def _config_show() -> None:
    """Show current configuration."""
    # Try to load from common locations
    config_paths = [
        Path.cwd() / "skills-config.yaml",
        Path.cwd() / "skills.yaml",
        Path.home() / ".config" / "skillengine" / "config.yaml",
    ]

    config = None
    loaded_from = None

    for path in config_paths:
        if path.exists():
            try:
                config = SkillsConfig.from_yaml(path)
                loaded_from = path
                break
            except Exception as e:
                console.print(f"[yellow]Failed to load {path}: {e}[/yellow]")

    if config is None:
        console.print("[dim]No config file found. Using defaults.[/dim]")
        config = SkillsConfig()
    else:
        console.print(f"[dim]Loaded from: {loaded_from}[/dim]\n")

    # Display configuration
    console.print("[bold]Current Configuration:[/bold]\n")
    console.print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))


def _config_init(output: str) -> None:
    """Initialize a new config file."""
    output_path = Path(output)

    if output_path.exists():
        console.print(f"[red]File already exists: {output_path}[/red]")
        sys.exit(1)

    # Create default config
    default_config = {
        "skill_dirs": ["./skills"],
        "watch": False,
        "watch_debounce_ms": 250,
        "default_timeout_seconds": 30.0,
        "prompt_format": "xml",
        "entries": {
            "# example-skill": {
                "enabled": True,
                "api_key": None,
                "env": {},
            }
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created config file: {output_path}[/green]")


def _config_path() -> None:
    """Show config file search paths."""
    console.print("[bold]Config file search paths:[/bold]\n")

    paths = [
        ("Current directory", Path.cwd() / "skills-config.yaml"),
        ("Current directory (alt)", Path.cwd() / "skills.yaml"),
        ("User config", Path.home() / ".config" / "skillengine" / "config.yaml"),
    ]

    for name, path in paths:
        exists = "[green]✓[/green]" if path.exists() else "[dim]·[/dim]"
        console.print(f"  {exists} {name}: {path}")


def cmd_ext(args: argparse.Namespace) -> None:
    """Extension management commands."""
    if args.ext_command == "list":
        _ext_list(args)
    elif args.ext_command == "info":
        _ext_info(args)
    else:
        console.print("[yellow]Usage: skills ext <list|info>[/yellow]")


def _ext_list(args: argparse.Namespace) -> None:
    """List installed extensions."""
    engine = _create_engine(getattr(args, "dirs", None))
    ext_manager = engine.init_extensions()

    extensions = ext_manager.get_extensions()
    if not extensions:
        console.print("[dim]No extensions installed.[/dim]")
        return

    table = Table(title="Installed Extensions")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Source", style="dim")
    table.add_column("Commands", style="green")
    table.add_column("Tools", style="yellow")

    # Build per-extension command/tool counts
    commands = ext_manager.get_commands()
    tools = ext_manager.get_tools()

    for ext in extensions:
        cmd_count = sum(1 for c in commands if c.extension_name == ext.name)
        tool_count = sum(1 for t in tools if t.extension_name == ext.name)
        table.add_row(
            ext.name,
            ext.version,
            ext.source,
            str(cmd_count),
            str(tool_count),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(extensions)} extensions[/dim]")


def _ext_info(args: argparse.Namespace) -> None:
    """Show extension details."""
    engine = _create_engine(getattr(args, "dirs", None))
    ext_manager = engine.init_extensions()

    ext = ext_manager.get_extension(args.name)
    if not ext:
        console.print(f"[red]Extension not found: {args.name}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]{ext.name}[/bold] v{ext.version}")
    if ext.description:
        console.print(f"[dim]{ext.description}[/dim]")
    console.print(f"  Source: {ext.source}")
    if ext.author:
        console.print(f"  Author: {ext.author}")

    # Show registered commands
    commands = [c for c in ext_manager.get_commands() if c.extension_name == ext.name]
    if commands:
        console.print("\n[bold]Commands:[/bold]")
        for cmd in commands:
            console.print(f"  /{cmd.name} - {cmd.description}")

    # Show registered tools
    tools = [t for t in ext_manager.get_tools() if t.extension_name == ext.name]
    if tools:
        console.print("\n[bold]Tools:[/bold]")
        for tool in tools:
            console.print(f"  {tool.name} - {tool.description}")


def cmd_reload(args: argparse.Namespace) -> None:
    """Reload skills and extensions."""
    engine = _create_engine(getattr(args, "dirs", None))
    engine.invalidate_cache()
    snapshot = engine.get_snapshot(force_reload=True)
    console.print(f"[green]Reloaded.[/green] {len(snapshot.skills)} skills available.")


def cmd_prompts(args: argparse.Namespace) -> None:
    """Prompt template management commands."""
    if args.prompts_command == "list":
        _prompts_list()
    elif args.prompts_command == "show":
        _prompts_show(args.name)
    else:
        console.print("[yellow]Usage: skills prompts <list|show>[/yellow]")


def _prompts_list() -> None:
    """List available prompt templates."""
    from skillengine.prompts import PromptTemplateLoader

    loader = PromptTemplateLoader()
    templates = loader.load_all()

    if not templates:
        console.print("[dim]No prompt templates found.[/dim]")
        console.print(
            "[dim]Place .md files in ~/.skillengine/prompts/ or ./.skillengine/prompts/[/dim]"
        )
        return

    table = Table(title="Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Variables", style="dim")
    table.add_column("Path", style="dim")

    for tmpl in templates:
        table.add_row(
            f"/{tmpl.name}",
            tmpl.description[:50],
            ", ".join(tmpl.variables) if tmpl.variables else "-",
            str(tmpl.file_path),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(templates)} templates[/dim]")


def _prompts_show(name: str) -> None:
    """Show a prompt template."""
    from skillengine.prompts import PromptTemplateLoader

    loader = PromptTemplateLoader()
    templates = loader.load_all()

    template = next((t for t in templates if t.name == name), None)
    if not template:
        console.print(f"[red]Template not found: {name}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]/{template.name}[/bold]")
    if template.description:
        console.print(f"[dim]{template.description}[/dim]")
    console.print(f"File: {template.file_path}")
    if template.variables:
        console.print(f"Variables: {', '.join(template.variables)}")
    console.print("\n[bold]Content:[/bold]")
    console.print(template.content)


def cmd_commands(args: argparse.Namespace) -> None:
    """List all slash commands from all sources."""
    from skillengine.commands import CommandRegistry
    from skillengine.prompts import PromptTemplateLoader

    engine = _create_engine(getattr(args, "dirs", None))

    # Initialize all command sources
    registry = CommandRegistry(engine)

    # Extensions
    ext_manager = engine.init_extensions()
    registry.sync_from_extensions(ext_manager)

    # Skills
    skills = engine.filter_skills()
    registry.sync_from_skills(skills)

    # Prompts
    prompt_loader = PromptTemplateLoader()
    templates = prompt_loader.load_all()
    registry.sync_from_prompts(templates, prompt_loader)

    # Display
    commands = registry.list_commands()

    table = Table(title="All Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for cmd in commands:
        # Skip quit aliases
        if cmd.name in ("/exit", "/q"):
            continue
        table.add_row(cmd.name, cmd.description[:60], cmd.source)

    console.print(table)
    console.print(f"\n[dim]Total: {len(commands)} commands[/dim]")


async def cmd_chat(args: argparse.Namespace) -> None:
    """Start interactive chat or run in a specific mode."""
    from skillengine.agent import AgentConfig, AgentRunner

    engine = _create_engine(getattr(args, "dirs", None))
    config_kwargs: dict[str, Any] = {}
    if args.model:
        config_kwargs["model"] = args.model

    agent_config = AgentConfig.from_env(**config_kwargs)
    agent = AgentRunner(engine, agent_config)

    mode = args.mode
    if mode == "json":
        from skillengine.modes.json_mode import JsonMode

        prompt_text = args.prompt_text
        if not prompt_text:
            console.print("[red]Error: JSON mode requires a prompt argument[/red]")
            sys.exit(1)
        json_mode = JsonMode()
        await json_mode.run(agent, prompt_text)
    elif mode == "rpc":
        from skillengine.modes.rpc_mode import RpcMode

        rpc_mode = RpcMode()
        await rpc_mode.run(agent)
    else:
        # Interactive mode (default)
        from skillengine.modes.interactive import InteractiveMode

        interactive = InteractiveMode()
        await interactive.run(agent)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the web UI server."""
    from skillengine.agent import AgentConfig, AgentRunner

    engine = _create_engine(getattr(args, "dirs", None))
    agent_config = AgentConfig.from_env()
    agent = AgentRunner(engine, agent_config)

    try:
        from skillengine.web.server import run_server

        console.print(f"[green]Starting web UI at http://{args.host}:{args.port}[/green]")
        run_server(agent=agent, host=args.host, port=args.port)
    except ImportError:
        console.print(
            "[red]Web UI requires the 'web' extra. Install with: pip install skillengine[web][/red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
