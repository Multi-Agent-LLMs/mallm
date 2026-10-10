import logging
import os
import math
import re
import time
from collections.abc import Iterator
from typing import Any, Optional, Union, cast

logger = logging.getLogger("mallm")

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from openai import APIError, APIConnectionError, RateLimitError, OpenAI

class Chat(LLM):    # type: ignore
    """A custom chat model that queries the chat API of HuggingFace Text Generation Inference

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    client: Any
    timeout: int = 120
    model: str = "gpt-3.5-turbo"
    stop_tokens: list[str] = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|reserved_special_token",
    ]
    max_tokens: int = 1024

    @staticmethod
    def _context_limited_max_tokens(error: Exception, current_max_tokens: int) -> Optional[int]:
        """Shrink max_tokens only when the API reports a context-window overflow."""
        message = str(error)
        max_ctx_match = re.search(r"maximum context length is (\d+) tokens", message)
        if not max_ctx_match:
            return None

        max_context_tokens = int(max_ctx_match.group(1))
        input_tokens = None
        for pattern in (
            r"parameter=input_tokens,\s*value=(\d+)",
            r"contains at least (\d+) input tokens",
            r"prompt_tokens=(\d+)",
        ):
            input_tokens_match = re.search(pattern, message)
            if input_tokens_match:
                input_tokens = int(input_tokens_match.group(1))
                break
        if input_tokens is None:
            return None

        # Leave a small buffer because the API reports "at least" input tokens.
        available_output_tokens = max_context_tokens - input_tokens - 32
        if available_output_tokens <= 0:
            return None
        if available_output_tokens >= current_max_tokens:
            return None
        return max(16, available_output_tokens)

    # Overwrite to send direct chat structure to tgi endpoint
    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        return cast(PromptValue, input)

    # Overwrite to send direct chat structure to tgi endpoint
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: Optional[list[str]] = None,
        callbacks: Optional[Union[Callbacks, list[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # this is a wrong cast, but we need it because we use a custom call function which can handle this
        return self.generate(prompts, stop=stop, callbacks=callbacks, **kwargs)

    @staticmethod
    def merge_consecutive_messages(
        messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if not messages:
            return []
        
        merged_messages = []
        current_role = messages[0]["role"]
        current_content = ""

        for msg in messages:
            if msg["role"] == current_role:
                current_content += msg["content"] + "\n\n"
            else:
                merged_messages.append(
                    {"role": current_role, "content": current_content.strip()}
                )
                current_role = msg["role"]
                current_content = msg["content"] + "\n\n"

        if current_content:
            merged_messages.append(
                {"role": current_role, "content": current_content.strip()}
            )

        return merged_messages

    def _call(  # type: ignore
        self,
        prompt,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        merged_messages = self.merge_consecutive_messages(prompt)
        retries = 0
        request_max_tokens = self.max_tokens
        while retries < 5:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=merged_messages,
                    stream=True,
                    stop=self.stop_tokens,
                    max_tokens=request_max_tokens,
                    logprobs=True,
                )
                # iterate and print stream
                collected_messages = []
                collected_reasoning = []
                log_prob_sum = 0.0
                log_prob_steps = 0
                for message in chat_completion:
                    delta = message.choices[0].delta
                    content_str = getattr(delta, "content", None)
                    reasoning_str = getattr(delta, "reasoning", None) or getattr(
                        delta, "reasoning_content", None
                    )
                    lp = message.choices[0].logprobs
                    if lp is not None:
                        content_lps = getattr(lp, "content", None)
                        if content_lps:
                            log_prob_sum += content_lps[0].logprob
                            log_prob_steps += 1
                    if content_str and content_str not in self.stop_tokens:
                        collected_messages.append(content_str)
                    if reasoning_str and reasoning_str not in self.stop_tokens:
                        collected_reasoning.append(reasoning_str)
                # Reasoning-parser servers (vLLM --reasoning-parser) stream the thinking as
                # delta.reasoning / reasoning_content. Keep only the answer channel so that
                # peers and the answer extraction see the final answer, not the thinking.
                # Fallback: if the model produced no answer text at all (e.g. thinking used
                # the whole token budget), use the reasoning text. MALLM_INCLUDE_REASONING=1
                # restores the old behaviour (thinking + answer concatenated).
                # vLLM's glm45 streaming parser leaks the last thinking tokens plus the closing
                # tag into the content channel ("...nal Solution: 4</think>The answer ..."):
                # the answer is whatever follows the LAST "</think>".
                joined = "".join(collected_messages)
                if "</think>" in joined:
                    pre, _, post = joined.rpartition("</think>")
                    collected_reasoning = collected_reasoning + [pre]
                    collected_messages = [post] if post else []
                if os.environ.get("MALLM_INCLUDE_REASONING", "").strip().lower() in ("1", "true", "yes"):
                    collected_messages = collected_reasoning + collected_messages
                elif not collected_messages:
                    joined_reasoning = "".join(collected_reasoning)
                    if "</think>" in joined_reasoning:
                        joined_reasoning = joined_reasoning.rpartition("</think>")[2]
                    collected_messages = [joined_reasoning] if joined_reasoning else collected_reasoning
                denom = len(collected_messages)
                if denom == 0 and log_prob_steps > 0:
                    denom = log_prob_steps
                if denom > 0:
                    log_prob_sum /= denom
                else:
                    log_prob_sum = float("-inf")
                break
            except (APIError, APIConnectionError, RateLimitError) as e:
                # Handle API error here, e.g. retry or log
                adjusted_max_tokens = self._context_limited_max_tokens(e, request_max_tokens)
                if adjusted_max_tokens is not None and adjusted_max_tokens < request_max_tokens:
                    logger.warning(
                        "Reducing max_tokens from %s to %s to fit the model context window.",
                        request_max_tokens,
                        adjusted_max_tokens,
                    )
                    request_max_tokens = adjusted_max_tokens
                    # Retry immediately with the adjusted token budget.
                    continue
                retries += 1
                if retries < 5:
                    logger.warning(
                        f"API returned an Error: {e}. Retry number {retries}..."
                    )
                    time.sleep(3)
                else:
                    logger.error(f" {e}: Exceeded maximum retries. This sample failed.")
                    raise Exception("Exceeded maximum API retries.")
        confidence = math.exp(log_prob_sum)
        if "confidence_callback" in kwargs:
            kwargs["confidence_callback"](confidence)
        return "".join(collected_messages)

    def _stream(  # type: ignore
        self,
        prompt,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator:  # type: ignore
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            stream=True,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
        )
        # iterate and print stream
        for message in chat_completion:
            yield message

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
