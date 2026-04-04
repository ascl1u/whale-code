from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.trajectories import (
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)

from agent.whale.parsing import parse_native_tool_calls
from agent.whale.types import ImageReadRequest, ToolCallResult


class WhaleLoopMixin:
    def _append_user_assistant_tool_round(
        self, chat: Chat, user_content: str, tool_response: ToolCallResult
    ) -> None:
        assistant_message = {"role": "assistant", "content": tool_response.content}
        if tool_response.tool_calls:
            assistant_message["tool_calls"] = tool_response.tool_calls
        chat._messages.append({"role": "user", "content": user_content})
        chat._messages.append(assistant_message)
        if tool_response.tool_calls:
            for tc in tool_response.tool_calls:
                chat._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "executed",
                    }
                )
            chat.reset_response_chain()
        if tool_response.usage:
            chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
            chat._cumulative_output_tokens += tool_response.usage.completion_tokens
            chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
            chat._cumulative_cost += tool_response.usage.cost_usd

    def _append_assistant_tool_round_only(
        self, chat: Chat, tool_response: ToolCallResult
    ) -> None:
        assistant_message = {"role": "assistant", "content": tool_response.content}
        if tool_response.tool_calls:
            assistant_message["tool_calls"] = tool_response.tool_calls
        chat._messages.append(assistant_message)
        if tool_response.tool_calls:
            for tc in tool_response.tool_calls:
                chat._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": "executed",
                    }
                )
            chat.reset_response_chain()
        if tool_response.usage:
            chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
            chat._cumulative_output_tokens += tool_response.usage.completion_tokens
            chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
            chat._cumulative_cost += tool_response.usage.cost_usd

    async def _timed_call_llm_with_tools(
        self, messages: list[dict]
    ) -> ToolCallResult:
        start_time = time.time()
        tool_response = await self._call_llm_with_tools(messages)
        self._api_request_times.append((time.time() - start_time) * 1000)
        return tool_response

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list[Command], bool, str, str, str, LLMResponse, ImageReadRequest | None]:
        _, prompt_path, response_path = logging_paths
        if prompt_path is not None:
            prompt_path.write_text(prompt, encoding="utf-8")

        messages = chat.messages.copy()
        messages.append({"role": "user", "content": prompt})

        try:
            tool_response = await self._timed_call_llm_with_tools(messages)
            self._append_user_assistant_tool_round(chat, prompt, tool_response)

        except ContextLengthExceededError:
            if not self._enable_summarize:
                raise
            if session is None:
                raise RuntimeError("Cannot summarize without session")

            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)
            summary_prompt = None
            try:
                summary_prompt, subagent_refs = await self._with_block_timeout(
                    self._summarize(chat, original_instruction, session)
                )
                self._pending_subagent_refs = subagent_refs
                self._pending_handoff_prompt = summary_prompt
            except Exception:
                pass

            if summary_prompt is None:
                current_screen = await self._with_block_timeout(
                    session.capture_pane(capture_entire=False)
                )
                limited_screen = current_screen[-1000:] if current_screen else ""
                summary_prompt = f"{original_instruction}\n\nCurrent state: {limited_screen}"

            messages = chat.messages.copy()
            messages.append({"role": "user", "content": summary_prompt})
            tool_response = await self._timed_call_llm_with_tools(messages)
            self._append_user_assistant_tool_round(
                chat, summary_prompt, tool_response
            )

        except OutputLengthExceededError:
            error_msg = (
                "ERROR!! Your response was truncated. "
                "Please provide a shorter response with fewer commands."
            )
            chat._messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "[truncated]"},
                    {"role": "user", "content": error_msg},
                ]
            )
            chat.reset_response_chain()
            tool_response = await self._timed_call_llm_with_tools(
                chat.messages.copy()
            )
            self._append_assistant_tool_round_only(chat, tool_response)

        if response_path is not None:
            response_path.write_text(
                "Content: "
                + (tool_response.content or "")
                + "\n\nTool Calls: "
                + json.dumps(tool_response.tool_calls, indent=2),
                encoding="utf-8",
            )

        commands, is_task_complete, feedback, analysis, plan, image_read = (
            parse_native_tool_calls(tool_response.tool_calls, self.logger)
        )
        llm_response = LLMResponse(
            content=tool_response.content or "",
            reasoning_content=tool_response.reasoning_content,
            usage=tool_response.usage,
        )
        return (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
            image_read,
        )

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
            if not await self._with_block_timeout(self._session.is_session_alive()):
                return episode + 1

            if original_instruction and self._enable_summarize:
                proactive_summary_result = await self._with_block_timeout(
                    self._check_proactive_summarization(
                        chat, original_instruction, self._session
                    )
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt

            logging_paths = self._setup_episode_logging(logging_dir, episode)
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
                image_read,
            ) = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )

            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
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

            if self._save_raw_content_in_trajectory:
                message_content = llm_response.content
            else:
                parts = []
                if analysis:
                    parts.append(f"Analysis: {analysis}")
                if plan:
                    parts.append(f"Plan: {plan}")
                message_content = "\n".join(parts) if parts else ""

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode}: {len(commands)} commands"
                + (" (image_read)" if image_read else "")
            )

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper {self._get_error_response_type()}."
                )
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[ObservationResult(content=prompt)]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=(
                                chat.total_cache_tokens - tokens_before_cache
                            )
                            or None,
                            cost_usd=(chat.total_cost - cost_before) or None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            if image_read is not None:
                image_read_result = await self._execute_image_read(
                    image_read, chat, original_instruction
                )
                was_pending_completion = self._pending_completion
                if is_task_complete:
                    if self._pending_completion:
                        observation = image_read_result
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            image_read_result
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{image_read_result}"
                        )
                    else:
                        observation = image_read_result

                tool_calls: list[ToolCall] | None = None
                observation_results: list[ObservationResult] = []
                if not self._save_raw_content_in_trajectory:
                    tool_calls_list: list[ToolCall] = [
                        ToolCall(
                            tool_call_id=f"call_{episode}_image_read",
                            function_name="image_read",
                            arguments={
                                "file_path": image_read.file_path,
                                "image_read_instruction": image_read.image_read_instruction,
                            },
                        )
                    ]
                    observation_results.append(ObservationResult(content=observation))
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )
                    tool_calls = tool_calls_list
                else:
                    observation_results.append(ObservationResult(content=observation))

                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=(
                                chat.total_cache_tokens - tokens_before_cache
                            )
                            or None,
                            cost_usd=(chat.total_cost - cost_before) or None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._dump_trajectory()
                if is_task_complete and was_pending_completion:
                    return episode + 1
                prompt = observation
                continue

            _, terminal_output = await self._with_block_timeout(
                self._execute_commands(commands, self._session)
            )
            was_pending_completion = self._pending_completion
            if is_task_complete:
                if self._pending_completion:
                    observation = terminal_output
                else:
                    self._pending_completion = True
                    observation = self._get_completion_confirmation_message(
                        terminal_output
                    )
            else:
                self._pending_completion = False
                if feedback and "WARNINGS:" in feedback:
                    observation = (
                        f"Previous response had warnings:\n{feedback}\n\n"
                        f"{self._limit_output_length(terminal_output)}"
                    )
                else:
                    observation = self._limit_output_length(terminal_output)

            tool_calls = None
            observation_results = []
            if not self._save_raw_content_in_trajectory:
                tool_calls_list = []
                if commands:
                    for idx, cmd in enumerate(commands):
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_{idx + 1}",
                                function_name="bash_command",
                                arguments={
                                    "keystrokes": cmd.keystrokes,
                                    "duration": cmd.duration_sec,
                                },
                            )
                        )
                    observation_results.append(ObservationResult(content=observation))
                if is_task_complete:
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=f"call_{episode}_task_complete",
                            function_name="mark_task_complete",
                            arguments={},
                        )
                    )
                    if not commands:
                        observation_results.append(ObservationResult(content=observation))
                elif not commands:
                    observation_results.append(ObservationResult(content=observation))
                tool_calls = tool_calls_list or None
            else:
                observation_results.append(ObservationResult(content=observation))

            self._trajectory_steps.append(
                Step(
                    step_id=len(self._trajectory_steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=self._model_name,
                    message=message_content,
                    reasoning_content=llm_response.reasoning_content,
                    tool_calls=tool_calls,
                    observation=Observation(results=observation_results),
                    metrics=Metrics(
                        prompt_tokens=chat.total_input_tokens - tokens_before_input,
                        completion_tokens=chat.total_output_tokens - tokens_before_output,
                        cached_tokens=(
                            chat.total_cache_tokens - tokens_before_cache
                        )
                        or None,
                        cost_usd=(chat.total_cost - cost_before) or None,
                        prompt_token_ids=llm_response.prompt_token_ids,
                        completion_token_ids=llm_response.completion_token_ids,
                        logprobs=llm_response.logprobs,
                    ),
                )
            )
            self._dump_trajectory()
            if is_task_complete and was_pending_completion:
                return episode + 1
            prompt = observation

        return self._n_episodes
