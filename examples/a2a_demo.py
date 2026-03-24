"""Demo: Load sawzhang-skills and generate Agent Cards + Registry awareness."""

import sys
from pathlib import Path

from skillengine import SkillsConfig, SkillsEngine
from skillengine.a2a import AgentCard, AgentRegistry


def main():
    # Load skills from sawzhang-skills plugin
    skill_dir = Path.home() / "code/claude_code_skills_demo/plugins/sawzhang-skills/skills"
    if not skill_dir.exists():
        print(f"Skill directory not found: {skill_dir}")
        sys.exit(1)

    engine = SkillsEngine(config=SkillsConfig(skill_dirs=[skill_dir]))
    skills = engine.load_skills()

    print(f"Loaded {len(skills)} skills from {skill_dir}\n")

    # Register all skills in the Agent Registry
    registry = AgentRegistry()
    registry.register_skills(skills)

    print(f"Registered {registry.count} agents\n")

    # Show Agent Cards
    print("=" * 60)
    print("AGENT CARDS")
    print("=" * 60)
    for agent in registry.all():
        card = agent.card
        print(f"\n📋 {card.name} (v{card.version})")
        print(f"   {card.description}")
        if card.tags:
            print(f"   Tags: {', '.join(card.tags)}")
        if card.model:
            print(f"   Model: {card.model}")
        print(f"   Card JSON keys: {list(card.to_dict().keys())}")

    # Show system prompt awareness block
    print("\n" + "=" * 60)
    print("ORCHESTRATOR AWARENESS BLOCK")
    print("=" * 60)
    print(registry.awareness_prompt_block())

    # Test routing
    print("=" * 60)
    print("ROUTING TESTS")
    print("=" * 60)

    queries = [
        "帮我看看这条推文",
        "review mcp tools",
        "自动迭代优化",
        "搜Twitter上关于Claude的讨论",
        "学CCA domain1",
        "cooking recipes",  # should match nothing
    ]

    for q in queries:
        matches = registry.match(q, top_k=2)
        if matches:
            names = [m.card.name for m in matches]
            print(f"\n  '{q}' → {names}")
        else:
            print(f"\n  '{q}' → (no match)")


if __name__ == "__main__":
    main()
