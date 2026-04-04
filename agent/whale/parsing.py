from __future__ import annotations

import json
import logging
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Command

from agent.whale.types import ImageReadRequest


def parse_native_tool_calls(
    tool_calls: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[list[Command], bool, str, str, str, ImageReadRequest | None]:
    commands: list[Command] = []
    is_task_complete = False
    feedback = ""
    analysis = ""
    plan = ""
    image_read = None

    if not tool_calls:
        feedback = (
            "WARNINGS: Your response contained no tool calls. "
            "Please use execute_commands to run commands."
        )
        return commands, is_task_complete, feedback, analysis, plan, image_read

    for tool_call in tool_calls:
        function_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        try:
            arguments = (
                json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool arguments: %s", arguments_str)
            continue

        if function_name == "execute_commands":
            analysis = arguments.get("analysis", "")
            plan = arguments.get("plan", "")
            for cmd in arguments.get("commands", []):
                commands.append(
                    Command(
                        keystrokes=cmd.get("keystrokes", ""),
                        duration_sec=min(cmd.get("duration", 1.0), 60),
                    )
                )
        elif function_name == "task_complete":
            is_task_complete = True
        elif function_name == "image_read":
            file_path = arguments.get("file_path", "")
            instruction = arguments.get("image_read_instruction", "")
            if file_path and instruction:
                image_read = ImageReadRequest(
                    file_path=file_path,
                    image_read_instruction=instruction,
                )
            else:
                feedback = (
                    "WARNINGS: image_read requires both file_path and "
                    "image_read_instruction arguments."
                )
        else:
            feedback = (
                f"WARNINGS: Unknown function '{function_name}'. "
                "Please use execute_commands, task_complete, or image_read."
            )
    return commands, is_task_complete, feedback, analysis, plan, image_read
