"""Eval case + dataset models with JSON / JSONL I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["EvalCase", "EvalDataset"]


@dataclass
class EvalCase:
    """A single evaluation case.

    Attributes:
        id: Stable identifier (used in reports).
        input: The input to feed the target (string or arbitrary value).
        expected: The expected outcome. Format is scorer-dependent —
            strings for text scorers, dicts for :class:`StructuredMatchScorer`,
            regex patterns for :class:`RegexScorer`, etc.
        tags: Free-form labels for filtering (e.g. ``["smoke", "slow"]``).
        metadata: Anything extra (skill name under test, expected tool calls).
    """

    id: str
    input: Any
    expected: Any = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            id=str(data["id"]),
            input=data.get("input"),
            expected=data.get("expected"),
            tags=list(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EvalDataset:
    """An ordered, named collection of :class:`EvalCase` objects."""

    name: str
    cases: list[EvalCase] = field(default_factory=list)
    description: str = ""

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def filter(
        self,
        *,
        tags: Iterable[str] | None = None,
        ids: Iterable[str] | None = None,
    ) -> EvalDataset:
        """Return a new dataset containing only cases that match the filters."""
        tag_set = set(tags) if tags else None
        id_set = set(ids) if ids else None
        out: list[EvalCase] = []
        for case in self.cases:
            if tag_set is not None and not (tag_set & set(case.tags)):
                continue
            if id_set is not None and case.id not in id_set:
                continue
            out.append(case)
        return EvalDataset(name=self.name, cases=out, description=self.description)

    # ----- I/O -----------------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        """Write each case as one JSON line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for case in self.cases:
                f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> EvalDataset:
        path = Path(path)
        cases: list[EvalCase] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cases.append(EvalCase.from_dict(json.loads(line)))
        return cls(name=name or path.stem, cases=cases)

    def to_json(self, path: str | Path) -> None:
        """Write the whole dataset (name + description + cases) as one JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "description": self.description,
                    "cases": [c.to_dict() for c in self.cases],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=str(data.get("name") or Path(path).stem),
            description=str(data.get("description") or ""),
            cases=[EvalCase.from_dict(c) for c in data.get("cases", [])],
        )
