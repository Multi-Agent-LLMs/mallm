from typing import Any, Iterator, Optional, Union, cast

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from openai import OpenAI


class Chat(LLM):
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

    client: OpenAI
    timeout: int = 120
    model: str = "gpt-3.5-turbo"
    stop_tokens: list[str] = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|reserved_special_token",
    ]
    max_tokens: int = 1024

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
        return self.generate(prompts, stop=stop, callbacks=callbacks, **kwargs)  # type: ignore

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
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            stream=True,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
        )
        # iterate and print stream
        collected_messages = []
        for message in chat_completion:
            message_str = message.choices[0].delta.content
            if message_str and message_str not in self.stop_tokens:
                collected_messages.append(message_str)

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