"""Append-only changelog writer for SkillOptimizer runs."""

from __future__ import annotations

from pathlib import Path

from skillengine.optimizer.models import MutationRecord, OptimizationReport


class ChangelogWriter:
    """Writes structured entries to OPTIMIZER_CHANGELOG.md inside the skill directory.

    Each optimizer run prepends a new H2 section to the file so the most
    recent run appears first. Within a run, rounds are appended in order.
    """

    def __init__(
        self,
        skill_dir: Path,
        filename: str = "OPTIMIZER_CHANGELOG.md",
    ) -> None:
        self.path = skill_dir / filename
        self._run_lines: list[str] = []

    def write_header(
        self,
        skill_name: str,
        checklist: list[str],
        test_input_count: int,
        timestamp: str,
    ) -> None:
        """Start a new run section. Called once before any rounds."""
        self._run_lines = [
            f"## Optimization Run — {timestamp}\n",
            f"**Skill:** {skill_name}  \n",
            f"**Checklist:** {len(checklist)} criteria  \n",
            f"**Test inputs:** {test_input_count}  \n",
            "\n",
            "| Round | Mutation | Baseline | Candidate | Accepted |\n",
            "|-------|----------|----------|-----------|----------|\n",
        ]

    def append_round(self, record: MutationRecord) -> None:
        """Append one mutation round as a table row."""
        desc = record.mutation_description.replace("|", "\\|")
        accepted_str = "yes" if record.accepted else "no"
        self._run_lines.append(
            f"| {record.round_number}"
            f" | {desc}"
            f" | {record.baseline_score:.2f}"
            f" | {record.candidate_mean:.2f}"
            f" | {accepted_str} |\n"
        )

    def write_footer(self, report: OptimizationReport) -> None:
        """Append summary and flush the complete run section to disk."""
        if report.converged:
            converged_str = "yes"
        else:
            converged_str = f"no (reached max_rounds={report.rounds_run})"
        accepted_count = len(report.accepted_mutations)
        total_count = len(report.mutations)

        self._run_lines += [
            "\n",
            f"**Initial score:** {report.initial_score:.2f}  \n",
            f"**Final score:** {report.final_score:.2f}  \n",
            f"**Converged:** {converged_str}  \n",
            f"**Accepted mutations:** {accepted_count} / {total_count}  \n",
            "\n---\n\n",
        ]

        new_section = "".join(self._run_lines)

        # Prepend new section so most recent run appears first
        if self.path.exists():
            existing = self.path.read_text(encoding="utf-8")
            self.path.write_text(new_section + existing, encoding="utf-8")
        else:
            self.path.write_text(new_section, encoding="utf-8")

        self._run_lines = []
