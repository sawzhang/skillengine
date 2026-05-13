"""``python -m skillengine.mcp`` — run a SkillEngine MCP server over stdio.

Usage::

    python -m skillengine.mcp --skill-dir ./skills
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ..config import SkillsConfig
from ..engine import SkillsEngine
from .server import serve_stdio


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m skillengine.mcp",
        description="Expose SkillEngine skills as an MCP server over stdio.",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory to load skills from (repeatable).",
    )
    parser.add_argument("--name", default="skillengine", help="Server name advertised to clients.")
    parser.add_argument("--version", default="0.3", help="Server version advertised to clients.")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional 'instructions' string sent in the initialize response.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    skill_dirs = args.skill_dir or [Path("./skills")]
    engine = SkillsEngine(SkillsConfig(skill_dirs=skill_dirs))
    asyncio.run(
        serve_stdio(
            engine=engine,
            name=args.name,
            version=args.version,
            instructions=args.instructions,
        )
    )


if __name__ == "__main__":
    main()
