"""Built-in eval suites bundled with SkillEngine.

The ``skill-dsl`` suite is a regression battery for the skill DSL pipeline —
YAML frontmatter parsing, ``$ARGUMENTS`` substitution, requires-based
filtering, and ``MarkdownSkillLoader`` edge cases. The target for this suite
is a small synchronous function (see :func:`run_skill_dsl_target`), so the
suite is self-contained and does not require an LLM to execute.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .dataset import EvalCase, EvalDataset
from .scorers import ContainsScorer, ExactMatchScorer, Scorer

__all__ = ["builtin_suite", "list_builtin_suites", "run_skill_dsl_target"]


# ---------------------------------------------------------------------------
# Target: a tiny pure-Python re-implementation of the surface area we test.
# ---------------------------------------------------------------------------


_ARG_PATTERN = re.compile(r"\$\{?(\d+|ARGUMENTS)\}?")


def _substitute_args(template: str, arguments: str) -> str:
    """Replicates ``AgentRunner._substitute_arguments`` minus env lookups."""
    parts = arguments.split()

    def repl(m: re.Match[str]) -> str:
        token = m.group(1)
        if token == "ARGUMENTS":
            return arguments
        idx = int(token) - 1
        return parts[idx] if 0 <= idx < len(parts) else ""

    return _ARG_PATTERN.sub(repl, template)


def _validate_name(name: str) -> str:
    if not name:
        return "error:empty-name"
    if len(name) > 64:
        return "error:name-too-long"
    if name.startswith("-"):
        return "error:leading-hyphen"
    if not re.fullmatch(r"[a-z0-9-]+", name):
        return "error:invalid-chars"
    return "ok"


def _validate_description(desc: str) -> str:
    if not desc:
        return "error:empty-description"
    if len(desc) > 1024:
        return "error:description-too-long"
    return "ok"


def _parse_frontmatter(text: str) -> dict[str, Any] | str:
    """Very small YAML-frontmatter parser (string scalars + simple lists)."""
    if not text.startswith("---\n"):
        return "error:no-frontmatter"
    end = text.find("\n---", 4)
    if end == -1:
        return "error:unterminated-frontmatter"
    body = text[4:end]
    out: dict[str, Any] = {}
    for line in body.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            return f"error:bad-line:{line!r}"
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
            out[key] = items
        else:
            out[key] = value.strip("\"'")
    return out


def run_skill_dsl_target(case_input: Any) -> str:
    """Target callable used by the ``skill-dsl`` built-in suite.

    Each case's ``input`` is a dict ``{"op": "...", ...args}``. The function
    dispatches to a small handler and returns a string. Scorers compare that
    string to the case's ``expected``.
    """
    if not isinstance(case_input, dict):
        return f"error:bad-input:{type(case_input).__name__}"
    op = case_input.get("op")
    if op == "substitute":
        return _substitute_args(case_input["template"], case_input["arguments"])
    if op == "validate_name":
        return _validate_name(case_input["name"])
    if op == "validate_description":
        return _validate_description(case_input["description"])
    if op == "parse_frontmatter":
        parsed = _parse_frontmatter(case_input["text"])
        if isinstance(parsed, str):
            return parsed
        # Stable string form (sorted keys) so ExactMatchScorer works.
        return ";".join(f"{k}={parsed[k]}" for k in sorted(parsed))
    return f"error:unknown-op:{op}"


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------


def _skill_dsl_dataset() -> EvalDataset:
    cases: list[EvalCase] = []

    # --- $ARGUMENTS / positional substitution (10 cases) -------------------
    sub_cases: list[tuple[str, str, str, str]] = [
        ("sub-arguments", "Hello $ARGUMENTS", "world", "Hello world"),
        ("sub-arg1-arg2", "$1 + $2", "alpha beta", "alpha + beta"),
        ("sub-braced", "Hello ${1}!", "name", "Hello name!"),
        ("sub-missing", "$1 - $5", "only-one", "only-one - "),
        ("sub-empty", "before $ARGUMENTS after", "", "before  after"),
        ("sub-repeated", "$1 $1 $1", "x", "x x x"),
        ("sub-three", "$3-$2-$1", "a b c", "c-b-a"),
        ("sub-no-token", "no tokens here", "anything", "no tokens here"),
        (
            "sub-mixed",
            "user=$1 query=$ARGUMENTS",
            "alice find me a doc",
            "user=alice query=alice find me a doc",
        ),
        ("sub-special-chars", "search:$ARGUMENTS", "a b c!?", "search:a b c!?"),
    ]
    for cid, template, args, expected in sub_cases:
        cases.append(
            EvalCase(
                id=cid,
                input={"op": "substitute", "template": template, "arguments": args},
                expected=expected,
                tags=["substitution"],
            )
        )

    # --- Name validation (10 cases) ----------------------------------------
    name_cases: list[tuple[str, str, str]] = [
        ("name-ok-simple", "hello", "ok"),
        ("name-ok-hyphens", "my-skill", "ok"),
        ("name-ok-digits", "skill-123", "ok"),
        ("name-empty", "", "error:empty-name"),
        ("name-leading-hyphen", "-skill", "error:leading-hyphen"),
        ("name-uppercase", "MySkill", "error:invalid-chars"),
        ("name-spaces", "my skill", "error:invalid-chars"),
        ("name-underscore", "my_skill", "error:invalid-chars"),
        ("name-too-long", "x" * 65, "error:name-too-long"),
        ("name-boundary-64", "x" * 64, "ok"),
    ]
    for cid, name, expected in name_cases:
        cases.append(
            EvalCase(
                id=cid,
                input={"op": "validate_name", "name": name},
                expected=expected,
                tags=["validation", "name"],
            )
        )

    # --- Description validation (5 cases) ----------------------------------
    desc_cases: list[tuple[str, str, str]] = [
        ("desc-ok", "Greet the user.", "ok"),
        ("desc-empty", "", "error:empty-description"),
        ("desc-too-long", "x" * 1025, "error:description-too-long"),
        ("desc-boundary-1024", "x" * 1024, "ok"),
        ("desc-multiline-ok", "A skill that\nspans lines", "ok"),
    ]
    for cid, desc, expected in desc_cases:
        cases.append(
            EvalCase(
                id=cid,
                input={"op": "validate_description", "description": desc},
                expected=expected,
                tags=["validation", "description"],
            )
        )

    # --- Frontmatter parsing (8 cases) -------------------------------------
    fm_cases: list[tuple[str, str, str]] = [
        (
            "fm-basic",
            "---\nname: hello\ndescription: Greet\n---\nBody.",
            "description=Greet;name=hello",
        ),
        (
            "fm-quoted-string",
            "---\nname: \"hello\"\ndescription: 'Hi there'\n---\nBody.",
            "description=Hi there;name=hello",
        ),
        (
            "fm-list-bins",
            "---\nname: x\nrequires_bins: [git, jq]\n---\nx",
            "name=x;requires_bins=['git', 'jq']",
        ),
        ("fm-missing", "Body only", "error:no-frontmatter"),
        ("fm-unterminated", "---\nname: x\nno terminator", "error:unterminated-frontmatter"),
        (
            "fm-comments",
            "---\n# comment\nname: c\ndescription: d\n---\nx",
            "description=d;name=c",
        ),
        (
            "fm-empty-list",
            "---\nname: e\nrequires_bins: []\n---\nx",
            "name=e;requires_bins=[]",
        ),
        ("fm-bad-line", "---\nnocolon\n---\nx", "error:bad-line:'nocolon'"),
    ]
    for cid, text, expected in fm_cases:
        cases.append(
            EvalCase(
                id=cid,
                input={"op": "parse_frontmatter", "text": text},
                expected=expected,
                tags=["frontmatter"],
            )
        )

    return EvalDataset(
        name="skill-dsl",
        description="Regression cases for the skill DSL: substitution, "
        "name/description validation, and frontmatter parsing.",
        cases=cases,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SuiteFactory = Callable[[], tuple[EvalDataset, list[Scorer], Callable[[Any], Any]]]


def _skill_dsl_suite() -> tuple[EvalDataset, list[Scorer], Callable[[Any], Any]]:
    return (
        _skill_dsl_dataset(),
        [ExactMatchScorer()],
        run_skill_dsl_target,
    )


def _smoke_suite() -> tuple[EvalDataset, list[Scorer], Callable[[Any], Any]]:
    """Trivial sanity suite — useful for verifying the runner wiring."""
    cases = [
        EvalCase(id="smoke-1", input="hello", expected="hello"),
        EvalCase(id="smoke-2", input="echo me", expected="echo me"),
    ]
    return (
        EvalDataset(name="smoke", description="Identity smoke test.", cases=cases),
        [ContainsScorer()],
        lambda x: x,
    )


_REGISTRY: dict[str, SuiteFactory] = {
    "skill-dsl": _skill_dsl_suite,
    "smoke": _smoke_suite,
}


def list_builtin_suites() -> list[str]:
    """Return the names of bundled eval suites."""
    return sorted(_REGISTRY)


def builtin_suite(
    name: str,
) -> tuple[EvalDataset, list[Scorer], Callable[[Any], Any]]:
    """Return ``(dataset, scorers, target)`` for the named built-in suite."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown built-in suite: {name!r}. Available: {list_builtin_suites()}")
    return _REGISTRY[name]()
