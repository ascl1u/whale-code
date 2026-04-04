from __future__ import annotations

import asyncio
import time

from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession

from agent.whale.types import BLOCK_TIMEOUT_SEC, MARKER_PREFIX, BlockError


class WhaleTerminalMixin:
    _INLINE_MARKER_FORBIDDEN_CHARS = frozenset("\"'`$&|<>();\\")

    async def _with_block_timeout(self, coro, timeout_sec: int = BLOCK_TIMEOUT_SEC):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            raise BlockError(f"Infrastructure API blocked for {timeout_sec}s") from exc

    def _can_inline_marker_poll(self, keystrokes: str) -> bool:
        if not keystrokes.endswith(("\n", "\r")):
            return False
        stripped = keystrokes.rstrip("\r\n")
        if not stripped:
            return False
        if "\n" in stripped or "\r" in stripped:
            return False
        return not any(
            char in self._INLINE_MARKER_FORBIDDEN_CHARS for char in stripped
        )

    def _build_inline_marker_command(self, keystrokes: str, marker: str) -> str:
        stripped = keystrokes.rstrip("\r\n")
        return f"{stripped}; printf '%s\\n' '{marker}'\n"

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        for command in commands:
            if not self._can_inline_marker_poll(command.keystrokes):
                await session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
                continue

            self._marker_seq += 1
            marker = f"{MARKER_PREFIX}{self._marker_seq}__"
            start = time.monotonic()

            await session.send_keys(
                self._build_inline_marker_command(command.keystrokes, marker),
                block=False,
                min_timeout_sec=0.0,
            )

            initial_sleep = min(0.05, command.duration_sec)
            if initial_sleep > 0:
                await asyncio.sleep(initial_sleep)

            marker_seen = False
            deadline = start + command.duration_sec
            while True:
                pane_content = await session.capture_pane()
                if marker in pane_content:
                    marker_seen = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.5, remaining))

            if marker_seen:
                saved = command.duration_sec - (time.monotonic() - start)
                if saved > 0.1:
                    self._total_time_saved += saved
                continue

            break

        output = await session.get_incremental_output()
        markers = {
            f"{MARKER_PREFIX}{seq}__" for seq in range(1, self._marker_seq + 1)
        }
        lines = output.split("\n")
        lines = [line for line in lines if not any(marker in line for marker in markers)]
        return False, self._limit_output_length("\n".join(lines))
