"""Interactive TUI mode for the coding agent."""

from __future__ import annotations

from typing import Any


class InteractiveMode:
    """Full TUI-based interactive coding agent.

    Uses TUI components for:
    - Markdown rendering of assistant responses
    - Syntax highlighting for code blocks
    - Tool execution display with streaming output
    - Session management UI
    - Model switching
    - Thinking level cycling
    """

    def __init__(self):
        self._agent = None
        self._running = False

    async def run(self, agent: Any) -> None:
        """Run the interactive TUI mode."""
        from rich.console import Console
        from rich.panel import Panel

        self._agent = agent
        self._running = True
        console = Console()

        # Print greeting
        console.print(
            Panel(
                "[bold]SkillEngine[/bold] - Interactive Mode\n"
                f"Model: {agent.config.model}\n"
                f"Skills: {len(agent.skills)}\n"
                "Type /help for commands, Ctrl+D to exit",
                title="Welcome",
                border_style="blue",
            )
        )

        while self._running:
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input == "/quit" or user_input == "/exit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input == "/help":
                console.print(
                    Panel(
                        "/quit, /exit - Exit\n"
                        "/clear - Clear history\n"
                        "/model - Show current model\n"
                        "/history - Show conversation history\n"
                        "/skills - List loaded skills",
                        title="Commands",
                    )
                )
                continue

            if user_input == "/clear":
                agent.clear_history()
                console.print("[dim]History cleared.[/dim]")
                continue

            if user_input == "/model":
                console.print(f"[dim]Current model: {agent.config.model}[/dim]")
                continue

            if user_input == "/skills":
                for s in agent.skills:
                    emoji = s.metadata.emoji or "\U0001f527"
                    console.print(f"  {emoji} {s.name} - {s.description[:60]}")
                continue

            if user_input == "/history":
                for msg in agent.get_history():
                    role_style = "green" if msg.role == "user" else "blue"
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    console.print(f"[{role_style}]{msg.role}:[/{role_style}] {content}")
                continue

            # Stream response
            console.print("[bold blue]Assistant:[/bold blue] ", end="")
            full_response = ""
            thinking = ""

            try:
                async for event in agent.chat_stream_events(user_input):
                    if event.type == "thinking_delta":
                        thinking += event.content
                    elif event.type == "text_delta":
                        console.print(event.content, end="", highlight=False)
                        full_response += event.content
                    elif event.type == "tool_call_start":
                        console.print(
                            f"\n[yellow]-> {event.tool_name}[/yellow]",
                            end="",
                        )
                    elif event.type == "tool_result":
                        result_preview = event.content[:200]
                        console.print(f"\n[dim]{result_preview}[/dim]")
                    elif event.type == "error":
                        console.print(f"\n[red]Error: {event.error}[/red]")

                if thinking:
                    console.print(f"\n[dim italic]Thinking: {thinking[:200]}...[/dim italic]")
                console.print()  # newline after response

            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    def stop(self) -> None:
        """Stop the interactive mode."""
        self._running = False
