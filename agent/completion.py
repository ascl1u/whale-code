"""Completion validation, task-aware verification planning, and persistent memory."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from harbor.environments.base import BaseEnvironment

MEMORY_PATH = Path(__file__).resolve().parent.parent / ".whale_memory.json"
_APP_PATH_PATTERN = re.compile(r"(/app/[^\s`\"'):,;]+)")
_TELNET_PATTERN = re.compile(r"telnet\s+([0-9A-Za-z._-]+)\s+([0-9]{2,5})")


@dataclass(frozen=True)
class VerificationRule:
    rule_id: str
    trigger_terms: tuple[str, ...]
    checklist: str
    guidance: str


@dataclass
class CompletionCheckResult:
    passed: bool
    feedback: str = ""
    task_complete_message: str = ""
    modes: tuple[str, ...] = ()


RULES: tuple[VerificationRule, ...] = (
    VerificationRule(
        rule_id="single_required_file",
        trigger_terms=(
            "single file",
            "single-file",
            "only file",
            "contained in a single file",
        ),
        checklist=(
            "Verify the deliverable directory contains only the required file and no "
            "temporary verification artifacts."
        ),
        guidance=(
            "If verification requires compiling or extracting temporary artifacts, place "
            "them outside the deliverable directory or remove them before task_complete."
        ),
    ),
    VerificationRule(
        rule_id="service_liveness",
        trigger_terms=(
            "connect via",
            "telnet",
            "port",
            "leave it running",
            "in the background",
            "ready",
        ),
        checklist=(
            "Verify the advertised endpoint or session is still reachable immediately "
            "before task_complete."
        ),
        guidance=(
            "Do not infer readiness from process startup alone. Re-check the actual "
            "endpoint, login prompt, or service response."
        ),
    ),
    VerificationRule(
        rule_id="compiled_path_execution",
        trigger_terms=(
            "compile",
            "gcc",
            "build",
            "cython",
            "extension",
            "from source",
        ),
        checklist=(
            "Execute the built artifact or changed code path directly during final verification."
        ),
        guidance=(
            "Do not rely only on imports, summaries, or partial smoke tests after build-like changes."
        ),
    ),
)

DEFAULT_MEMORY = {
    "rules": [asdict(rule) for rule in RULES],
    "observations": [],
}


def load_failure_memory() -> dict:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))

    MEMORY_PATH.write_text(
        json.dumps(DEFAULT_MEMORY, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json.loads(json.dumps(DEFAULT_MEMORY))


def save_failure_memory(memory: dict) -> None:
    MEMORY_PATH.write_text(
        json.dumps(memory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_instruction_paths(instruction: str) -> list[str]:
    seen: list[str] = []
    for match in _APP_PATH_PATTERN.findall(instruction):
        if match not in seen:
            seen.append(match)
    return seen


def _matching_rules(instruction: str) -> list[VerificationRule]:
    lowered = instruction.lower()
    matches: list[VerificationRule] = []
    for rule in RULES:
        if any(term in lowered for term in rule.trigger_terms):
            matches.append(rule)
    return matches


def infer_verification_modes(instruction: str) -> tuple[str, ...]:
    return tuple(rule.rule_id for rule in _matching_rules(instruction))


def build_memory_guidance(instruction: str, memory: dict) -> str:
    rules = _matching_rules(instruction)
    if not rules:
        return ""

    lines = ["Persistent memory:"]
    for rule in rules:
        lines.append(f"- {rule.guidance}")

    observations = memory.get("observations", [])
    for observation in observations[-12:]:
        rule_id = observation.get("rule_id", "")
        if any(rule.rule_id == rule_id for rule in rules):
            note = observation.get("note", "")
            count = observation.get("count", 1)
            if note:
                lines.append(f"- Previous failure pattern ({count}x): {note}")

    return "\n".join(lines)


def build_completion_checklist(instruction: str) -> str:
    rules = _matching_rules(instruction)
    if not rules:
        return ""
    return "\n".join(f"- {rule.checklist}" for rule in rules)


def record_completion_failure(
    memory: dict,
    *,
    rule_id: str,
    note: str,
) -> dict:
    observations = memory.setdefault("observations", [])
    for observation in observations:
        if observation.get("rule_id") == rule_id and observation.get("note") == note:
            observation["count"] = int(observation.get("count", 1)) + 1
            observation["last_seen_at"] = datetime.now(timezone.utc).isoformat()
            return memory

    observations.append(
        {
            "rule_id": rule_id,
            "note": note,
            "count": 1,
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return memory


async def run_pre_completion_check(
    environment: BaseEnvironment,
    *,
    user: str | int | None,
    instruction: str,
    terminal_output: str,
    memory: dict,
) -> CompletionCheckResult:
    modes = infer_verification_modes(instruction)

    if "single_required_file" in modes:
        paths = extract_instruction_paths(instruction)
        if len(paths) == 1:
            required_path = Path(paths[0])
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
                note = (
                    f"Expected only {required_name} in {parent}, found: {entries or ['<empty>']}"
                )
                record_completion_failure(
                    memory,
                    rule_id="single_required_file",
                    note=note,
                )
                save_failure_memory(memory)
                return CompletionCheckResult(
                    passed=False,
                    feedback=(
                        "Completion check failed: the final filesystem state does not "
                        f"match the task contract. Expected only `{required_name}` in "
                        f"`{parent}`, found: {entries or ['<empty>']}. Remove temporary "
                        "verification artifacts before task_complete."
                    ),
                    task_complete_message=(
                        "task_complete rejected: final filesystem state is wrong. "
                        f"Expected only `{required_name}` in `{parent}`, found: "
                        f"{entries or ['<empty>']}."
                    ),
                    modes=modes,
                )

    task_complete_message = "Completion check passed."
    checklist = build_completion_checklist(instruction)
    if checklist:
        task_complete_message = (
            "Completion check passed. Relevant verification modes:\n" + checklist
        )

    return CompletionCheckResult(
        passed=True,
        task_complete_message=task_complete_message,
        modes=modes,
    )
