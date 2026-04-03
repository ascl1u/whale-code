"""Prompt constants for the Whale agent."""


def completion_checklist(instruction: str, terminal_output: str) -> str:
    """Second task_complete: minimal contract; full state is already in tool + terminal context."""
    return (
        f"Original task:\n{instruction}\n\n"
        f"Current terminal (reference):\n{terminal_output}\n\n"
        "Grading will run next. Call task_complete again immediately if every requirement is met "
        "and you already ran any verification command the task describes (e.g. test.sh, pytest). "
        "Do not run celebratory or redundant commands. Use execute_commands only if the task is "
        "not yet verified or a specific requirement is still missing."
    )
