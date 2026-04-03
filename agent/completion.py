"""Generic completion validation and checklist generation."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from harbor.environments.base import BaseEnvironment

_APP_PATH_PATTERN = re.compile(r"(/app/[^\s`\"'):,;]+)")
_EXACT_ARTIFACT_TERMS = (
    "single file",
    "single-file",
    "only file",
    "contained in a single file",
)


@dataclass(frozen=True)
class CompletionConstraint:
    kind: str
    target_path: str
    checklist: str


@dataclass
class CompletionCheckResult:
    passed: bool
    feedback: str = ""
    task_complete_message: str = ""


def build_completion_checklist(instruction: str) -> str:
    constraints = infer_completion_constraints(instruction)
    if not constraints:
        return ""
    return "\n".join(f"- {constraint.checklist}" for constraint in constraints)


def infer_completion_constraints(instruction: str) -> tuple[CompletionConstraint, ...]:
    lowered = instruction.lower()
    if not any(term in lowered for term in _EXACT_ARTIFACT_TERMS):
        return ()

    paths = _extract_instruction_paths(instruction)
    if len(paths) != 1:
        return ()

    target_path = paths[0]
    return (
        CompletionConstraint(
            kind="exact_directory_contents",
            target_path=target_path,
            checklist=(
                "Verify the deliverable directory contains only the required artifact "
                "and no temporary verification outputs."
            ),
        ),
    )


async def run_pre_completion_check(
    environment: BaseEnvironment,
    *,
    user: str | int | None,
    instruction: str,
) -> CompletionCheckResult:
    constraints = infer_completion_constraints(instruction)

    for constraint in constraints:
        if constraint.kind != "exact_directory_contents":
            continue

        required_path = Path(constraint.target_path)
        parent = required_path.parent.as_posix()
        required_name = required_path.name
        command = (
            f"if [ -d {shlex.quote(parent)} ]; then "
            f"find {shlex.quote(parent)} -maxdepth 1 -mindepth 1 -printf '%f\\n' | sort; "
            f"else printf '__missing_dir__\\n'; fi"
        )
        result = await environment.exec(command=command, user=user)
        entries = [
            line.strip()
            for line in (result.stdout or "").splitlines()
            if line.strip()
        ]
        if entries != [required_name]:
            return CompletionCheckResult(
                passed=False,
                feedback=(
                    "Completion check failed: the final filesystem state does not "
                    f"match the task contract. Expected only `{required_name}` in "
                    f"`{parent}`, found: {entries or ['<empty>']}."
                ),
                task_complete_message=(
                    "task_complete rejected: final filesystem state is wrong. "
                    f"Expected only `{required_name}` in `{parent}`, found: "
                    f"{entries or ['<empty>']}."
                ),
            )

    if constraints:
        return CompletionCheckResult(
            passed=True,
            task_complete_message=(
                "Completion check passed. Relevant verification modes:\n"
                + build_completion_checklist(instruction)
            ),
        )

    return CompletionCheckResult(
        passed=True,
        task_complete_message="Completion check passed.",
    )


def _extract_instruction_paths(instruction: str) -> list[str]:
    seen: list[str] = []
    for match in _APP_PATH_PATTERN.findall(instruction):
        if match not in seen:
            seen.append(match)
    return seen
