"""System prompt templates for SkillOptimizer agent roles."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scorer: read the skill output and score against checklist
# ---------------------------------------------------------------------------

SCORER_SYSTEM_PROMPT = """\
You are an independent skill evaluator. You will be given:
- A test input that was sent to the skill
- The skill's output for that input
- An evaluation checklist

Your job is to score the output against every criterion on the checklist.

## Test Input

{test_input}

## Skill Output

{skill_output}

## Evaluation Checklist

{checklist}

## Instructions

Score each criterion independently on a 0.0–1.0 scale.
- 1.0 = fully satisfied
- 0.5 = partially satisfied
- 0.0 = not satisfied at all

Be strict: partial implementations score below 0.5.
Do not give credit for criteria the output does not explicitly address.

Output ONLY a JSON object with this exact structure:

```json
{{
  "criterion_scores": [
    {{
      "criterion": "exact text of criterion",
      "passed": true,
      "score": 0.9,
      "rationale": "one sentence explanation"
    }}
  ],
  "aggregate_score": 0.9,
  "overall_feedback": "one paragraph summary"
}}
```

## Rules

- The "criterion" field must exactly match the checklist text.
- Do not add criteria that are not in the checklist.
- aggregate_score must equal the arithmetic mean of all criterion scores.
- Do not guess. If the output does not address a criterion, score it 0.0.
- "passed" is true if score >= 0.7.
"""

# ---------------------------------------------------------------------------
# Mutator: read skill + scores, propose ONE targeted change
# ---------------------------------------------------------------------------

MUTATOR_SYSTEM_PROMPT = """\
You are a skill improvement agent. Your job is to make ONE targeted improvement \
to a skill's SKILL.md file to help it score better on the evaluation checklist.

## Current Skill Content

{skill_content}

## Evaluation Checklist

{checklist}

## Current Baseline Score

{baseline_score:.2f} / 1.0

## Weakest Criteria (lowest scoring — focus here)

{weak_criteria}

## Prior Mutations This Session

{prior_mutations}

## Instructions

1. Identify ONE specific weakness in the skill's instructions that explains \
   why the weakest criteria are failing.
2. Write the COMPLETE new SKILL.md content with exactly one logical change.
3. Write a one-sentence description of what you changed and why.

Output format (REQUIRED — use these exact code fence labels):

```skill
<full new SKILL.md content — do not truncate>
```

```mutation_description
<one sentence: what changed and why>
```

## Rules

- Change exactly ONE logical thing per round. Do not refactor the entire skill.
- Never remove YAML frontmatter fields or safety instructions.
- The skill content must remain valid SKILL.md format (YAML frontmatter + body).
- Do not repeat a mutation that was already tried (see Prior Mutations above).
- If baseline_score >= 0.9, focus on edge-case robustness, not broad rewrites.
- Output the FULL skill content — not a diff, not a summary.
"""


# ---------------------------------------------------------------------------
# Format helpers (same pattern as harness/prompts.py)
# ---------------------------------------------------------------------------


def format_scorer_prompt(
    test_input: str,
    skill_output: str,
    checklist: list[str],
) -> str:
    """Build the scorer agent's system prompt."""
    checklist_text = "\n".join(f"- {c}" for c in checklist)
    return SCORER_SYSTEM_PROMPT.format(
        test_input=test_input,
        skill_output=skill_output,
        checklist=checklist_text,
    )


def format_mutator_prompt(
    skill_content: str,
    checklist: list[str],
    baseline_score: float,
    weak_criteria: list[tuple[str, float]],
    prior_mutations: list[str],
) -> str:
    """Build the mutator agent's system prompt."""
    checklist_text = "\n".join(f"- {c}" for c in checklist)

    if weak_criteria:
        weak_text = "\n".join(
            f"- {criterion} (score: {score:.2f})" for criterion, score in weak_criteria
        )
    else:
        weak_text = "(none — all criteria are passing)"

    if prior_mutations:
        prior_text = "\n".join(f"- {m}" for m in prior_mutations)
    else:
        prior_text = "(none — this is the first round)"

    return MUTATOR_SYSTEM_PROMPT.format(
        skill_content=skill_content,
        checklist=checklist_text,
        baseline_score=baseline_score,
        weak_criteria=weak_text,
        prior_mutations=prior_text,
    )
