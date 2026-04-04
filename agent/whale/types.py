from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harbor.models.metric import UsageInfo


class BlockError(Exception):
    """Raised when an infrastructure call blocks for too long."""


BLOCK_TIMEOUT_SEC = 600
MARKER_PREFIX = "__CMDEND__"


@dataclass
class ToolCallResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    usage: UsageInfo | None = None


@dataclass
class ImageReadRequest:
    file_path: str
    image_read_instruction: str
