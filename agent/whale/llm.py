from __future__ import annotations

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

from harbor.llms.base import (
    ContextLengthExceededError,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.metric import UsageInfo

from agent.context import add_anthropic_caching
from agent.tools import TOOLS
from agent.whale.types import ImageReadRequest, ToolCallResult

LLM_REQUEST_TIMEOUT_SEC = 120


class WhaleLLMMixin:
    def _extract_tool_calls(self, response: object) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
        return tool_calls

    def _extract_usage_info(self, response: object) -> UsageInfo | None:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            pass
        return UsageInfo(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            cache_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cost_usd=cost,
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
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            drop_params=True,
        )

    async def _execute_image_read(
        self,
        image_read: ImageReadRequest,
        chat: Chat,
        original_instruction: str = "",
    ) -> str:
        if self._session is None:
            raise RuntimeError("Session is not set")

        result = await self._with_block_timeout(
            self._session.environment.exec(command=f"base64 {image_read.file_path}")
        )
        if result.return_code != 0:
            return (
                f"ERROR: Failed to read file '{image_read.file_path}': "
                f"{result.stderr or ''}"
            )

        b64 = (result.stdout or "").replace("\n", "")
        ext = Path(image_read.file_path).suffix.lower()
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
                f"ERROR: Unsupported image format '{ext}'. "
                "Convert to PNG first, then retry image_read."
            )

        messages = add_anthropic_caching(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_read.image_read_instruction},
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
            response = await self._call_llm_for_image(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._llm.get_model_output_limit(),
            )
        except Exception as exc:
            return f"ERROR: {exc}"

        response_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        if usage:
            chat._cumulative_input_tokens += usage.get("prompt_tokens", 0)
            chat._cumulative_output_tokens += usage.get("completion_tokens", 0)
            prompt_details = usage.get("prompt_tokens_details")
            cached = (
                getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            )
            chat._cumulative_cache_tokens += cached or 0

        return f"File Read Result for '{image_read.file_path}':\n{response_text}"

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
            "timeout": LLM_REQUEST_TIMEOUT_SEC,
            "drop_params": True,
        }
        if hasattr(self._llm, "_api_base") and self._llm._api_base:
            kwargs["api_base"] = self._llm._api_base
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
            kwargs["temperature"] = 1

        try:
            response = await litellm.acompletion(**kwargs)
        except LiteLLMContextWindowExceededError as exc:
            raise ContextLengthExceededError() from exc

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = self._extract_tool_calls(response)
        usage_info = self._extract_usage_info(response)

        if response.choices[0].finish_reason == "length":
            raise OutputLengthExceededError(
                "Response was truncated due to max tokens limit",
                truncated_response=content,
            )

        reasoning_content = getattr(message, "reasoning_content", None)
        return ToolCallResult(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_info,
        )
