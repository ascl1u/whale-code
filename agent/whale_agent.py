"""
WhaleAgent: Terminus2-based agent with native tool calling.
"""

from __future__ import annotations

from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.whale.llm import WhaleLLMMixin
from agent.whale.loop import WhaleLoopMixin
from agent.whale.terminal import WhaleTerminalMixin
from agent.whale.types import (
    BLOCK_TIMEOUT_SEC,
    BlockError,
    ImageReadRequest,
    ToolCallResult,
)

__all__ = [
    "BLOCK_TIMEOUT_SEC",
    "BlockError",
    "ImageReadRequest",
    "ToolCallResult",
    "WhaleAgent",
]


class WhaleAgent(WhaleLoopMixin, WhaleLLMMixin, WhaleTerminalMixin, Terminus2):
    """Native tool-calling agent on Terminus2."""

    SUPPORTS_ATIF = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker_seq = 0
        self._total_time_saved = 0.0

    @staticmethod
    def name() -> str:
        return "whale"

    def version(self) -> str | None:
        return "whale-baseline"

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        self._original_instruction = instruction
        await super().run(instruction, environment, context)

    def _get_parser(self):
        return None

    def _get_prompt_template_path(self) -> Path:
        return Path(__file__).parent / "prompt_template.txt"

    def _get_error_response_type(self) -> str:
        return "response with valid tool calls"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        instruction = getattr(self, "_original_instruction", "N/A")
        return (
            f"Original task:\n{instruction}\n\n"
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure you want to mark the task as complete?\n\n"
            "Do not run more commands just to double-check. If the checklist is satisfied, call task_complete again immediately. Only continue working if you found a concrete unmet requirement in the terminal state above.\n\n"
            "[!] Checklist\n"
            "- Does your solution meet the requirements in the original task above? [TODO/DONE]\n"
            "- Does your solution account for potential changes in numeric values, array sizes, file contents, or configuration parameters? [TODO/DONE]\n"
            "- Have you verified your solution from the all perspectives of a test engineer, a QA engineer, and the user who requested this task?\n"
            "  - test engineer [TODO/DONE]\n"
            "  - QA engineer [TODO/DONE]\n"
            "  - user who requested this task [TODO/DONE]\n\n"
            "After this point, solution grading will begin and no further edits will be possible. If everything looks good, call task_complete tool again."
        )
