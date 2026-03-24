#!/usr/bin/env python3
"""
SkillEngine Demo

Demonstrates automatic skill loading and agent execution
similar to Claude Code's experience.

Usage:
    # Run demo (auto-loads .env file)
    python examples/agent_demo.py

    # Or run interactive mode
    python examples/agent_demo.py --interactive

Note: The agent automatically loads .env file from the current
directory or parent directories. No need to manually source it.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skillengine import AgentRunner, create_agent


async def demo_basic():
    """Basic usage demo."""
    print("=" * 60)
    print("SkillEngine - Basic Demo")
    print("=" * 60)

    # Create agent with skills
    agent = await create_agent(
        skill_dirs=[Path(__file__).parent / "skills"],
        system_prompt="You are a helpful assistant with access to various skills.",
    )

    # Show loaded skills
    print(f"\nLoaded {len(agent.skills)} skills:")
    for skill in agent.skills:
        emoji = skill.metadata.emoji or "🔧"
        print(f"  {emoji} {skill.name}: {skill.description[:60]}...")

    print("\n" + "-" * 60)

    # Test a simple chat
    print("\nUser: What skills do you have available?")
    response = await agent.chat("What skills do you have available?")
    print(f"\nAssistant: {response.content}")

    return agent


async def demo_skill_invocation(agent: AgentRunner):
    """Demo user-invocable skills (slash commands)."""
    print("\n" + "=" * 60)
    print("Skill Invocation Demo (Slash Commands)")
    print("=" * 60)

    invocable = agent.user_invocable_skills
    print(f"\nUser-invocable skills: {[f'/{s.name}' for s in invocable]}")

    if invocable:
        skill = invocable[0]
        print(f"\nUser: /{skill.name} help")
        response = await agent.chat(f"/{skill.name} help", reset=True)
        print(f"\nAssistant: {response.content[:500]}...")


async def demo_tool_execution(agent: AgentRunner):
    """Demo automatic tool execution."""
    print("\n" + "=" * 60)
    print("Tool Execution Demo")
    print("=" * 60)

    print("\nUser: List the files in the current directory")
    response = await agent.chat("List the files in the current directory", reset=True)
    print(f"\nAssistant: {response.content}")


async def run_interactive():
    """Run interactive chat mode."""
    print("=" * 60)
    print("SkillEngine - Interactive Mode")
    print("=" * 60)

    agent = await create_agent(
        skill_dirs=[Path(__file__).parent / "skills"],
        system_prompt="You are a helpful assistant with access to various skills. "
                      "You can execute commands to help users accomplish tasks.",
        watch_skills=True,  # Enable hot-reload
    )

    await agent.run_interactive(
        prompt="You: ",
        greeting="\nWelcome to SkillEngine interactive mode!",
    )


async def main():
    """Main entry point."""
    if "--interactive" in sys.argv or "-i" in sys.argv:
        await run_interactive()
    else:
        agent = await demo_basic()
        await demo_skill_invocation(agent)
        await demo_tool_execution(agent)
        print("\n" + "=" * 60)
        print("Demo complete! Run with --interactive for chat mode.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
