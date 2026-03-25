"""Agent Card generation from Skills.

An Agent Card is a machine-readable description of an agent's capabilities,
following the A2A (Agent-to-Agent) protocol. In SkillEngine, Agent Cards are
derived automatically from SKILL.md definitions — the Skill is the single
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skillengine.models import Skill


@dataclass
class AgentCapabilities:
    """What the agent supports at the protocol level."""

    streaming: bool = False
    multi_turn: bool = False
    push_notifications: bool = False


@dataclass
class AgentCardSkill:
    """A single capability within an Agent Card."""

    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """A2A Agent Card derived from a SkillEngine Skill.

    Mapping from SKILL.md:
        skill.name          → card.name
        skill.description   → card.description / card.skills[0].description
        skill.metadata.*    → card.version, card.author, card.tags
        skill.a2a.*         → card.capabilities, card.skills[0].schemas
    """

    name: str
    description: str
    version: str = "1.0.0"
    url: str | None = None

    # Capabilities
    skills: list[AgentCardSkill] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)

    # Metadata
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    cost_hint: str | None = None
    model: str | None = None

    @classmethod
    def from_skill(cls, skill: Skill, base_url: str | None = None) -> AgentCard:
        """Generate an Agent Card from a Skill instance.

        Extracts all information from existing Skill fields.
        If the Skill has an `a2a` block in its frontmatter,
        those fields are used for schema and capability details.
        """
        # Extract a2a config from the skill's raw frontmatter
        # (SkillEntry stores this; Skill itself doesn't have it as a field,
        #  so we check for it via a convention)
        a2a = _extract_a2a_config(skill)

        card_skill = AgentCardSkill(
            name=skill.name,
            description=skill.description,
            input_schema=a2a.get("input_schema"),
            output_schema=a2a.get("output_schema"),
            tags=skill.metadata.tags,
        )

        return cls(
            name=skill.name,
            description=skill.description,
            version=skill.metadata.version or "1.0.0",
            url=f"{base_url}/agents/{skill.name}" if base_url else None,
            skills=[card_skill],
            capabilities=AgentCapabilities(
                streaming=a2a.get("streaming", False),
                multi_turn=a2a.get("stateful", False),
            ),
            author=skill.metadata.author,
            tags=skill.metadata.tags,
            cost_hint=a2a.get("cost_hint"),
            model=skill.model,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to A2A /.well-known/agent.json format."""
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "multi_turn": self.capabilities.multi_turn,
                "push_notifications": self.capabilities.push_notifications,
            },
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    **({"input_schema": s.input_schema} if s.input_schema else {}),
                    **({"output_schema": s.output_schema} if s.output_schema else {}),
                    **({"tags": s.tags} if s.tags else {}),
                }
                for s in self.skills
            ],
        }
        if self.url:
            d["url"] = self.url
        if self.author:
            d["author"] = self.author
        if self.tags:
            d["tags"] = self.tags
        if self.cost_hint:
            d["cost_hint"] = self.cost_hint
        if self.model:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Deserialize from A2A Agent Card JSON."""
        caps = data.get("capabilities", {})
        skills_data = data.get("skills", [])

        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            url=data.get("url"),
            input_modes=data.get("input_modes", ["text"]),
            output_modes=data.get("output_modes", ["text"]),
            capabilities=AgentCapabilities(
                streaming=caps.get("streaming", False),
                multi_turn=caps.get("multi_turn", False),
                push_notifications=caps.get("push_notifications", False),
            ),
            skills=[
                AgentCardSkill(
                    name=s["name"],
                    description=s["description"],
                    input_schema=s.get("input_schema"),
                    output_schema=s.get("output_schema"),
                    tags=s.get("tags", []),
                )
                for s in skills_data
            ],
            author=data.get("author"),
            tags=data.get("tags", []),
            cost_hint=data.get("cost_hint"),
            model=data.get("model"),
        )

    def to_embedding_text(self) -> str:
        """Generate text representation for semantic indexing.

        Used by the Orchestrator for embedding-based agent matching.
        Combines name, description, tags, and skill descriptions.
        """
        parts = [self.name, self.description]
        parts.extend(self.tags)
        for s in self.skills:
            if s.description != self.description:  # avoid duplication
                parts.append(s.description)
            parts.extend(s.tags)
        return " | ".join(filter(None, parts))

    def to_summary_line(self) -> str:
        """One-line summary for system prompt injection."""
        line = f"**{self.name}**: {self.description}"
        if self.tags:
            line += f" [{', '.join(self.tags)}]"
        return line


def _extract_a2a_config(skill: Skill) -> dict[str, Any]:
    """Extract a2a configuration from a Skill.

    The a2a config can be stored in:
    1. skill._a2a_config (set by loader when parsing frontmatter)
    2. Empty dict (default — skill has no a2a config)
    """
    return getattr(skill, "_a2a_config", {})
