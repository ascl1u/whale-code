"""Prompt constants for the Whale agent."""


def completion_checklist(instruction: str, terminal_output: str) -> str:
    """Completion reminder for strict final verification before ending the rollout."""
    return (
        f"Original task:\n{instruction}\n\n"
        f"Current terminal (reference):\n{terminal_output}\n\n"
        "Grading will run next. Before task_complete, verify the task's exact deliverable from the "
        "terminal output: the required file contents, command result, or system state. Run any "
        "task-described test or verification command. If you changed code, builds, or compiled "
        "extensions, execute the changed path directly. Do not rely only on imports, partial smoke "
        "tests, or a summary of what you think changed. Use execute_commands only if a specific "
        "requirement is still unverified or missing."
    )
