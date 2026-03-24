#!/usr/bin/env python3
"""
Test each skill's execution capability.

Tests:
1. PDF - pypdf text extraction
2. Algorithmic Art - p5.js template loading
3. PPTX - markitdown text extraction
4. Slack GIF Creator - PIL/Pillow availability
5. Web Artifacts Builder - script availability
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skillengine import create_agent


async def test_skill(agent, skill_name: str, prompt: str, description: str):
    """Test a single skill."""
    print(f"\n{'='*60}")
    print(f"Testing: {skill_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    try:
        response = await agent.chat(prompt, reset=True)
        print(f"\n✅ Response:\n{response.content[:500]}...")

        # Check if tools were executed
        history = agent.get_history()
        tool_calls = [m for m in history if m.role == 'tool']
        if tool_calls:
            print(f"\n📦 Tools executed: {len(tool_calls)}")
            for tc in tool_calls[:3]:  # Show first 3
                output = tc.content[:200] + "..." if len(tc.content) > 200 else tc.content
                print(f"   - {tc.name}: {output}")

        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    print("=" * 60)
    print("SkillEngine - Skill Execution Tests")
    print("=" * 60)

    # Create agent
    agent = await create_agent(
        skill_dirs=[Path(__file__).parent / "skills"],
        system_prompt="You are a helpful assistant. Execute commands to test skill capabilities. Be concise.",
    )

    print(f"\nLoaded {len(agent.skills)} skills:")
    for s in agent.skills:
        emoji = s.metadata.emoji or "🔧"
        print(f"  {emoji} {s.name}")

    results = {}

    # Test 1: PDF Skill
    results["pdf"] = await test_skill(
        agent,
        "pdf",
        "Check if pypdf is installed by running: python -c \"from pypdf import PdfReader; print('pypdf OK')\"",
        "Test PDF processing capability"
    )

    # Test 2: Algorithmic Art Skill
    results["algorithmic-art"] = await test_skill(
        agent,
        "algorithmic-art",
        "Check if the algorithmic art template exists by running: ls -la examples/skills/algorithmic-art/templates/",
        "Test algorithmic art template availability"
    )

    # Test 3: PPTX Skill
    results["pptx"] = await test_skill(
        agent,
        "pptx",
        "Check if markitdown is available by running: python -c \"import markitdown; print('markitdown OK')\"",
        "Test PPTX processing capability"
    )

    # Test 4: Slack GIF Creator
    results["slack-gif-creator"] = await test_skill(
        agent,
        "slack-gif-creator",
        "Check if PIL/Pillow is installed by running: python -c \"from PIL import Image, ImageDraw; print('PIL OK')\"",
        "Test GIF creation capability"
    )

    # Test 5: Web Artifacts Builder
    results["web-artifacts-builder"] = await test_skill(
        agent,
        "web-artifacts-builder",
        "Check if the init script exists by running: ls -la examples/skills/web-artifacts-builder/scripts/",
        "Test web artifacts scripts availability"
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for skill, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {skill}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
