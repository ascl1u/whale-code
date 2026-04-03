"""General command execution policy for terminal safety."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_INTERACTIVE_ROOT_COMMANDS = {
    "ftp",
    "irb",
    "less",
    "lua",
    "man",
    "more",
    "mysql",
    "nano",
    "nc",
    "netcat",
    "node",
    "psql",
    "python",
    "python3",
    "redis-cli",
    "sftp",
    "sh",
    "socat",
    "sqlite3",
    "ssh",
    "tail",
    "telnet",
    "top",
    "vi",
    "vim",
    "watch",
}
_SHELL_CONTROL_TOKENS = {"|", "||", "&&", ";", "&"}


@dataclass(frozen=True)
class CommandPolicy:
    mode: Literal["foreground", "background", "isolated"]
    root_command: str | None = None


def classify_command_submission(keystrokes: str) -> CommandPolicy:
    stripped = keystrokes.rstrip()
    if not stripped:
        return CommandPolicy(mode="foreground")

    if stripped.endswith("&"):
        return CommandPolicy(mode="background")

    root_command = _extract_root_command(stripped.rstrip("\r\n"))
    if root_command in _INTERACTIVE_ROOT_COMMANDS:
        return CommandPolicy(mode="isolated", root_command=root_command)

    return CommandPolicy(mode="foreground", root_command=root_command)


def _extract_root_command(command: str) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None

    if not tokens:
        return None

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SHELL_CONTROL_TOKENS:
            return None
        if token == "env":
            index += 1
            while index < len(tokens) and _looks_like_env_assignment(tokens[index]):
                index += 1
            continue
        if token == "timeout":
            index = _skip_timeout_wrapper(tokens, index + 1)
            continue
        if token == "stdbuf":
            index += 1
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue
        if token in {"command", "nohup"}:
            index += 1
            continue
        return Path(token).name.lower()

    return None


def _skip_timeout_wrapper(tokens: list[str], index: int) -> int:
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option in {"-k", "--kill-after", "-s", "--signal"} and index < len(tokens):
            index += 1
    if index < len(tokens):
        index += 1
    return index


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _ = token.split("=", 1)
    return bool(name) and name.replace("_", "").isalnum()
