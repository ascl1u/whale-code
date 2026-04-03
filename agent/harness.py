"""Small, explicit helpers for chat/tool protocol (no fake tool payloads)."""

from __future__ import annotations

import shlex
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat

# After tool results, observations live in tool messages — avoid duplicating them in a huge user turn.
CONTINUATION_PROMPT = "Continue with your next action using the tool results above."
COMPLETION_CONFIRMATION_PROMPT = (
    "If the task is fully verified, call task_complete again now. "
    "Do not run more commands unless the tool results above show a specific gap."
)


async def discover_verifier_hint(environment: BaseEnvironment, *, timeout_sec: int = 30) -> str:
    """Best-effort: common TB entrypoints under /app or cwd. No task-specific logic."""
    inner = (
        "cd /app 2>/dev/null || cd . 2>/dev/null || true; "
        'for f in test.sh tests/test.sh run_tests.sh verify.sh check.sh; do '
        'if [ -f "$f" ]; then '
        'echo "If the task allows, verify with: bash $f or ./$f (from the task working directory)."; '
        "break; "
        "fi; "
        "done"
    )
    cmd = f"bash -lc {shlex.quote(inner)}"
    try:
        r = await environment.exec(cmd, timeout_sec=timeout_sec)
    except Exception:
        return ""
    out = (r.stdout or "").strip()
    return out if out else ""


def append_assistant_turn(chat: Chat, user_content: str, result: Any) -> None:
    """Append user + assistant (with optional tool_calls). Does not append tool role rows."""
    assistant: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        assistant["tool_calls"] = result.tool_calls
    chat._messages.append({"role": "user", "content": user_content})
    chat._messages.append(assistant)
    chat.reset_response_chain()
    if result.usage:
        chat._cumulative_input_tokens += result.usage.prompt_tokens
        chat._cumulative_output_tokens += result.usage.completion_tokens
        chat._cumulative_cache_tokens += result.usage.cache_tokens
        chat._cumulative_cost += result.usage.cost_usd


def append_tool_result_messages(
    chat: Chat,
    tool_calls: list[dict[str, Any]],
    content_by_id: dict[str, str],
) -> None:
    """One tool message per tool_call_id, in declaration order."""
    for tc in tool_calls:
        tid = tc.get("id") or ""
        body = content_by_id.get(tid)
        if body is None:
            body = "(no result recorded for this tool call)"
        chat._messages.append({"role": "tool", "tool_call_id": tid, "content": body})
    chat.reset_response_chain()


def build_tool_content_map(
    tool_calls: list[dict[str, Any]],
    *,
    terminal_text: str,
    read_results: dict[str, str],
    image_results: dict[str, str],
    task_complete_by_id: dict[str, str],
) -> dict[str, str]:
    """Map tool_call_id -> faithful string result for each tool in this assistant message."""
    out: dict[str, str] = {}
    for tc in tool_calls:
        tid = tc.get("id") or ""
        fn = tc.get("function", {}).get("name", "")
        if fn == "execute_commands":
            out[tid] = terminal_text
        elif fn == "read_file":
            out[tid] = read_results.get(tid, "ERROR: read_file was not executed.")
        elif fn == "image_read":
            out[tid] = image_results.get(tid, "ERROR: image_read was not executed.")
        elif fn == "task_complete":
            out[tid] = task_complete_by_id.get(tid, "task_complete received.")
        else:
            out[tid] = f"(unhandled tool: {fn})"
    return out
