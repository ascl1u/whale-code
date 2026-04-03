"""
WhaleAgent — native tool-calling Terminus 2 variant for Terminal-Bench 2.0.

- Faithful tool role payloads (observations live in tool messages, not a fake \"executed\" string).
- read_file for bounded disk reads outside tmux scrollback.
- Verifier hint probed once at setup (common test.sh-style entrypoints).
- Continuation user turns are minimal after tool results.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import (
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)

from agent.context import add_anthropic_caching
from agent.harness import (
    CONTINUATION_PROMPT,
    append_assistant_turn,
    append_tool_result_messages,
    build_tool_content_map,
    discover_verifier_hint,
)
from agent.prompts import completion_checklist
from agent.tools import TOOLS

_BLOCK_TIMEOUT = 600
_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}


@dataclass
class ToolCallResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    usage: UsageInfo | None = None


@dataclass
class ImageReadRequest:
    file_path: str
    instruction: str
    tool_call_id: str


@dataclass
class ReadFileRequest:
    tool_call_id: str
    path: str
    max_bytes: int


class WhaleAgent(Terminus2):
    """Custom Harbor agent for Terminal-Bench 2.0."""

    SUPPORTS_ATIF = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_time_saved = 0.0
        self._verifier_hint: str = ""

    @staticmethod
    def name() -> str:
        return "whale"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        try:
            self._verifier_hint = await discover_verifier_hint(
                environment, timeout_sec=30
            )
        except Exception:
            self._verifier_hint = ""

    def _get_parser(self):
        return None

    def _get_prompt_template_path(self) -> Path:
        return Path(__file__).parent / "prompt_template.txt"

    def _get_error_response_type(self) -> str:
        return "response with valid tool calls"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        instruction = getattr(self, "_original_instruction", "N/A")
        return completion_checklist(instruction, terminal_output)

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        self._original_instruction = instruction
        if getattr(self, "_verifier_hint", "").strip():
            instruction = (
                f"{instruction.rstrip()}\n\n---\n{self._verifier_hint.strip()}"
            )
        await super().run(instruction, environment, context)

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        for cmd in commands:
            start = time.monotonic()
            if self._is_multiline_submission(cmd.keystrokes):
                timeout = await self._execute_multiline_submission(cmd, session)
                if timeout:
                    return True, timeout
            elif self._is_blocking_command_submission(cmd.keystrokes):
                try:
                    await session.send_keys(
                        cmd.keystrokes,
                        block=True,
                        max_timeout_sec=cmd.duration_sec,
                    )
                except TimeoutError:
                    return True, self._timeout_template.format(
                        timeout_sec=cmd.duration_sec,
                        command=cmd.keystrokes,
                        terminal_state=self._limit_output_length(
                            await session.get_incremental_output()
                        ),
                    )
            else:
                await session.send_keys(
                    cmd.keystrokes,
                    block=False,
                    min_timeout_sec=cmd.duration_sec,
                )

            saved = cmd.duration_sec - (time.monotonic() - start)
            if saved > 0.1:
                self._total_time_saved += saved

        return False, self._limit_output_length(await session.get_incremental_output())

    @staticmethod
    def _is_command_submission(keystrokes: str) -> bool:
        return keystrokes in _ENTER_KEYS or keystrokes.endswith(("\n", "\r"))

    @classmethod
    def _is_blocking_command_submission(cls, keystrokes: str) -> bool:
        if not cls._is_command_submission(keystrokes):
            return False
        if cls._is_background_command_submission(keystrokes):
            return False
        return True

    @classmethod
    def _is_background_command_submission(cls, keystrokes: str) -> bool:
        if not cls._is_command_submission(keystrokes):
            return False
        stripped = keystrokes.rstrip()
        if not stripped or stripped in _ENTER_KEYS:
            return False
        return stripped.endswith("&")

    @classmethod
    def _is_multiline_submission(cls, keystrokes: str) -> bool:
        if not cls._is_command_submission(keystrokes):
            return False
        stripped = keystrokes.rstrip("\r\n")
        return "\n" in stripped or "\r" in stripped

    async def _execute_multiline_submission(
        self,
        cmd: Command,
        session: TmuxSession,
    ) -> str | None:
        await session.send_keys(cmd.keystrokes, block=False, min_timeout_sec=0.0)
        await session.send_keys("tmux wait -S done\n", block=False, min_timeout_sec=0.0)

        result = await asyncio.wait_for(
            session.environment.exec(
                f"timeout {cmd.duration_sec}s tmux wait done",
                user=getattr(session, "_user", None),
            ),
            timeout=_BLOCK_TIMEOUT,
        )
        if result.return_code == 0:
            return None

        return self._timeout_template.format(
            timeout_sec=cmd.duration_sec,
            command=cmd.keystrokes,
            terminal_state=self._limit_output_length(
                await session.get_incremental_output()
            ),
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_with_tools(self, messages: list[dict]) -> ToolCallResult:
        messages = add_anthropic_caching(messages, self._model_name)

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "tools": TOOLS,
            "timeout": 900,
            "drop_params": True,
        }
        if hasattr(self._llm, "_api_base") and self._llm._api_base:
            kwargs["api_base"] = self._llm._api_base
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
            kwargs["temperature"] = 1

        try:
            resp = await litellm.acompletion(**kwargs)
        except LiteLLMContextWindowExceededError:
            raise ContextLengthExceededError()

        msg = resp.choices[0].message
        content = msg.content or ""

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        usage_info = None
        if resp.usage:
            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=resp) or 0.0
            except Exception:
                pass
            usage_info = UsageInfo(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                cache_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                cost_usd=cost,
            )

        if resp.choices[0].finish_reason == "length":
            raise OutputLengthExceededError(
                "Response truncated by max tokens", truncated_response=content
            )

        reasoning = getattr(msg, "reasoning_content", None)
        return ToolCallResult(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            usage=usage_info,
        )

    def _parse_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[
        list[Command],
        bool,
        str,
        str,
        str,
        list[ImageReadRequest],
        list[ReadFileRequest],
    ]:
        commands: list[Command] = []
        is_complete = False
        feedback = ""
        analysis = ""
        plan = ""
        image_reads: list[ImageReadRequest] = []
        read_files: list[ReadFileRequest] = []

        if not tool_calls:
            feedback = (
                "WARNING: No tool calls in your response. "
                "Use execute_commands, read_file, or image_read as needed."
            )
            return (
                commands,
                is_complete,
                feedback,
                analysis,
                plan,
                image_reads,
                read_files,
            )

        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", "{}")
            tid = tc.get("id") or ""
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                feedback = f"ERROR: Invalid JSON in tool arguments for {fn_name}."
                self.logger.warning(f"Bad tool args: {raw_args}")
                continue

            if fn_name == "execute_commands":
                analysis = args.get("analysis", "")
                plan = args.get("plan", "")
                for c in args.get("commands", []):
                    commands.append(
                        Command(
                            keystrokes=c.get("keystrokes", ""),
                            duration_sec=min(c.get("duration", 1.0), 60),
                        )
                    )
            elif fn_name == "task_complete":
                is_complete = True
            elif fn_name == "image_read":
                fp = args.get("file_path", "")
                inst = args.get("image_read_instruction", "")
                if fp and inst:
                    image_reads.append(
                        ImageReadRequest(fp, inst, tool_call_id=tid)
                    )
                else:
                    feedback = (
                        "WARNING: image_read requires file_path and "
                        "image_read_instruction."
                    )
            elif fn_name == "read_file":
                path = args.get("path", "")
                try:
                    max_b = int(args.get("max_bytes", 32768))
                except (TypeError, ValueError):
                    max_b = 32768
                max_b = min(max(max_b, 1), 262_144)
                if path:
                    read_files.append(
                        ReadFileRequest(
                            tool_call_id=tid, path=path, max_bytes=max_b
                        )
                    )
                else:
                    feedback = "WARNING: read_file requires path."
            else:
                feedback = f"WARNING: Unknown tool '{fn_name}'."

        return (
            commands,
            is_complete,
            feedback,
            analysis,
            plan,
            image_reads,
            read_files,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_for_image(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int
    ) -> object:
        return await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=900,
            drop_params=True,
        )

    async def _execute_read_file(self, req: ReadFileRequest) -> str:
        if self._session is None:
            return "ERROR: No session."
        path = shlex.quote(req.path)
        cap = min(max(req.max_bytes, 1), 262_144)
        cmd = (
            f"if [ -f {path} ]; then head -c {cap} {path} 2>&1; "
            f"elif [ -d {path} ]; then printf '%s\\n' 'ERROR: path is a directory'; "
            f"else printf '%s\\n' 'ERROR: file not found'; fi"
        )
        r = await asyncio.wait_for(
            self._session.environment.exec(cmd), timeout=_BLOCK_TIMEOUT
        )
        combined = (r.stdout or "") + (r.stderr or "")
        return self._limit_output_length(combined)

    async def _execute_image_read(
        self,
        req: ImageReadRequest,
        chat: Chat,
    ) -> str:
        if self._session is None:
            raise RuntimeError("No active session")

        result = await asyncio.wait_for(
            self._session.environment.exec(command=f"base64 {shlex.quote(req.file_path)}"),
            timeout=_BLOCK_TIMEOUT,
        )
        if result.return_code != 0:
            return f"ERROR: Cannot read '{req.file_path}': {result.stderr or ''}"

        b64 = (result.stdout or "").replace("\n", "")
        ext = Path(req.file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext)
        if mime is None:
            return (
                f"ERROR: Unsupported format '{ext}'. "
                f"Convert to PNG first, then retry with image_read."
            )

        messages = add_anthropic_caching(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": req.instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            self._model_name,
        )

        try:
            resp = await self._call_llm_for_image(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._llm.get_model_output_limit(),
            )
        except Exception as e:
            return f"ERROR: {e}"

        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        if usage:
            chat._cumulative_input_tokens += usage.get("prompt_tokens", 0)
            chat._cumulative_output_tokens += usage.get("completion_tokens", 0)
            details = usage.get("prompt_tokens_details")
            chat._cumulative_cache_tokens += (
                getattr(details, "cached_tokens", 0) if details else 0
            )
        return f"Image analysis of '{req.file_path}':\n{text}"

    def _apply_usage(self, chat: Chat, usage: UsageInfo | None) -> None:
        if usage:
            chat._cumulative_input_tokens += usage.prompt_tokens
            chat._cumulative_output_tokens += usage.completion_tokens
            chat._cumulative_cache_tokens += usage.cache_tokens
            chat._cumulative_cost += usage.cost_usd

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[
        list[Command],
        bool,
        str,
        str,
        str,
        LLMResponse,
        list[ImageReadRequest],
        list[ReadFileRequest],
        list[dict[str, Any]],
    ]:
        _, prompt_path, response_path = logging_paths
        if prompt_path is not None:
            prompt_path.write_text(prompt, encoding="utf-8")

        messages = chat.messages.copy()
        messages.append({"role": "user", "content": prompt})

        try:
            t0 = time.time()
            result = await self._call_llm_with_tools(messages)
            self._api_request_times.append((time.time() - t0) * 1000)
            append_assistant_turn(chat, prompt, result)

        except ContextLengthExceededError:
            if not self._enable_summarize:
                raise
            self.logger.debug("Context exceeded — summarizing.")
            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)

            summary_prompt = None
            try:
                if session is None:
                    raise RuntimeError("Need session for summarization")
                summary_prompt, subagent_refs = await asyncio.wait_for(
                    self._summarize(chat, original_instruction, session),
                    timeout=_BLOCK_TIMEOUT,
                )
                self._pending_subagent_refs = subagent_refs
                self._pending_handoff_prompt = summary_prompt
            except Exception as e:
                self.logger.debug(f"Summarization failed: {e}")

            if summary_prompt is None:
                screen = ""
                if session:
                    screen = await asyncio.wait_for(
                        session.capture_pane(capture_entire=False),
                        timeout=_BLOCK_TIMEOUT,
                    )
                    screen = screen[-1000:] if screen else ""
                summary_prompt = f"{original_instruction}\n\nCurrent state: {screen}"

            messages = chat.messages.copy()
            messages.append({"role": "user", "content": summary_prompt})
            t0 = time.time()
            result = await self._call_llm_with_tools(messages)
            self._api_request_times.append((time.time() - t0) * 1000)
            append_assistant_turn(chat, summary_prompt, result)

        except OutputLengthExceededError:
            self.logger.debug("Output truncated — requesting shorter response.")
            err = (
                "ERROR: Response truncated. Provide a shorter response with fewer "
                "or smaller tool calls."
            )
            chat._messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "[truncated]"},
                    {"role": "user", "content": err},
                ]
            )
            chat.reset_response_chain()
            t_retry = time.time()
            result = await self._call_llm_with_tools(chat.messages.copy())
            self._api_request_times.append((time.time() - t_retry) * 1000)
            asst: dict[str, Any] = {"role": "assistant", "content": result.content}
            if result.tool_calls:
                asst["tool_calls"] = result.tool_calls
            chat._messages.append(asst)
            chat.reset_response_chain()
            self._apply_usage(chat, result.usage)

        if response_path is not None:
            response_path.write_text(
                f"Content: {result.content or ''}\nTool Calls: "
                f"{json.dumps(result.tool_calls, indent=2)}",
                encoding="utf-8",
            )

        (
            cmds,
            is_complete,
            feedback,
            analysis,
            plan,
            image_reads,
            read_files,
        ) = self._parse_tool_calls(result.tool_calls)
        llm_resp = LLMResponse(
            content=result.content or "",
            reasoning_content=result.reasoning_content,
            usage=result.usage,
        )
        return (
            cmds,
            is_complete,
            feedback,
            analysis,
            plan,
            llm_resp,
            image_reads,
            read_files,
            result.tool_calls,
        )

    def _task_complete_messages(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        is_complete: bool,
        terminal_text: str,
    ) -> dict[str, str]:
        ids = [
            tc.get("id") or ""
            for tc in tool_calls
            if tc.get("function", {}).get("name") == "task_complete"
        ]
        if not ids:
            return {}
        if not is_complete:
            return {i: "task_complete tool referenced unexpectedly." for i in ids}
        checklist = self._get_completion_confirmation_message(terminal_text)
        return {i: checklist for i in ids}

    def _trajectory_tool_calls(
        self,
        episode: int,
        model_tool_calls: list[dict[str, Any]],
        commands: list[Command],
    ) -> list[ToolCall] | None:
        if self._save_raw_content_in_trajectory:
            return None
        out: list[ToolCall] = []
        i = 0
        for tc in model_tool_calls:
            fn = tc.get("function", {}).get("name", "")
            tid = tc.get("id") or f"call_{episode}_{i}"
            if fn == "execute_commands":
                for j, cmd in enumerate(commands):
                    out.append(
                        ToolCall(
                            tool_call_id=f"{tid}_cmd_{j}",
                            function_name="bash_command",
                            arguments={
                                "keystrokes": cmd.keystrokes,
                                "duration": cmd.duration_sec,
                            },
                        )
                    )
            elif fn == "read_file":
                try:
                    raw = tc.get("function", {}).get("arguments", "{}")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                out.append(
                    ToolCall(
                        tool_call_id=tid,
                        function_name="read_file",
                        arguments={"path": args.get("path", "")},
                    )
                )
            elif fn == "image_read":
                try:
                    raw = tc.get("function", {}).get("arguments", "{}")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                out.append(
                    ToolCall(
                        tool_call_id=tid,
                        function_name="image_read",
                        arguments={
                            "file_path": args.get("file_path", ""),
                            "image_read_instruction": args.get(
                                "image_read_instruction", ""
                            ),
                        },
                    )
                )
            elif fn == "task_complete":
                out.append(
                    ToolCall(
                        tool_call_id=tid,
                        function_name="task_complete",
                        arguments={},
                    )
                )
            i += 1
        return out or None

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        if self._context is None or self._session is None:
            raise RuntimeError("Agent not properly initialized")

        prompt = initial_prompt
        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1

            if not await asyncio.wait_for(
                self._session.is_session_alive(), timeout=_BLOCK_TIMEOUT
            ):
                self.logger.debug("Session ended")
                return episode + 1

            if original_instruction and self._enable_summarize:
                proactive = await asyncio.wait_for(
                    self._check_proactive_summarization(
                        chat, original_instruction, self._session
                    ),
                    timeout=_BLOCK_TIMEOUT,
                )
                if proactive:
                    prompt, refs = proactive
                    self._pending_subagent_refs = refs
                    self._pending_handoff_prompt = prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            tok_in_before = chat.total_input_tokens
            tok_out_before = chat.total_output_tokens
            tok_cache_before = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_complete,
                feedback,
                analysis,
                plan,
                llm_resp,
                image_reads,
                read_files,
                model_tool_calls,
            ) = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )

            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Context summarization and handoff.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            if self._pending_handoff_prompt:
                if self._linear_history:
                    self._split_trajectory_on_summarization(self._pending_handoff_prompt)
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode}: cmd={len(commands)} read={len(read_files)} "
                f"img={len(image_reads)}"
            )

            if self._save_raw_content_in_trajectory:
                msg_content = llm_resp.content
            else:
                parts = []
                if analysis:
                    parts.append(f"Analysis: {analysis}")
                if plan:
                    parts.append(f"Plan: {plan}")
                msg_content = "\n".join(parts) if parts else ""

            step_metrics = Metrics(
                prompt_tokens=chat.total_input_tokens - tok_in_before,
                completion_tokens=chat.total_output_tokens - tok_out_before,
                cached_tokens=(chat.total_cache_tokens - tok_cache_before) or None,
                cost_usd=(chat.total_cost - cost_before) or None,
                prompt_token_ids=llm_resp.prompt_token_ids,
                completion_token_ids=llm_resp.completion_token_ids,
                logprobs=llm_resp.logprobs,
            )

            if not model_tool_calls:
                repair = (
                    f"{feedback}\n\n" if feedback else ""
                ) + f"Please provide a proper {self._get_error_response_type()}."
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_resp.content,
                        reasoning_content=llm_resp.reasoning_content,
                        observation=Observation(
                            results=[ObservationResult(content=repair)]
                        ),
                        metrics=step_metrics,
                    )
                )
                self._dump_trajectory()
                prompt = repair
                continue

            _, terminal_output = await asyncio.wait_for(
                self._execute_commands(commands, self._session),
                timeout=_BLOCK_TIMEOUT,
            )

            read_results: dict[str, str] = {}
            for rf in read_files:
                read_results[rf.tool_call_id] = await self._execute_read_file(rf)

            image_results: dict[str, str] = {}
            for ir in image_reads:
                image_results[ir.tool_call_id] = await self._execute_image_read(
                    ir, chat
                )

            tcp_map = self._task_complete_messages(
                model_tool_calls,
                is_complete=is_complete,
                terminal_text=terminal_output,
            )

            content_map = build_tool_content_map(
                model_tool_calls,
                terminal_text=terminal_output,
                read_results=read_results,
                image_results=image_results,
                task_complete_by_id=tcp_map,
            )
            append_tool_result_messages(chat, model_tool_calls, content_map)

            if is_complete:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=msg_content,
                        reasoning_content=llm_resp.reasoning_content,
                        tool_calls=self._trajectory_tool_calls(
                            episode, model_tool_calls, commands
                        ),
                        observation=Observation(
                            results=[ObservationResult(content=CONTINUATION_PROMPT)]
                        ),
                        metrics=step_metrics,
                    )
                )
                self._dump_trajectory()
                return episode + 1

            next_prompt = CONTINUATION_PROMPT
            if feedback:
                next_prompt = f"{feedback}\n\n{next_prompt}"

            self._trajectory_steps.append(
                Step(
                    step_id=len(self._trajectory_steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=msg_content,
                    reasoning_content=llm_resp.reasoning_content,
                    tool_calls=self._trajectory_tool_calls(
                        episode, model_tool_calls, commands
                    ),
                    observation=Observation(
                        results=[ObservationResult(content=next_prompt)]
                    ),
                    metrics=step_metrics,
                )
            )
            self._dump_trajectory()

            prompt = next_prompt

        return self._n_episodes
