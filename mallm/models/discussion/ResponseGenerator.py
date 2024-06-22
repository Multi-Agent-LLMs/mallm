from abc import ABC, abstractmethod
from typing import Any

from mallm.models.Chat import Chat


class ResponseGenerator(ABC):
    @abstractmethod
    def __init__(self, llm: Chat):
        pass

    @abstractmethod
    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        chain_of_thought: bool,
        baseline: bool,
        drafting: bool,
    ) -> dict[str, Any]:
        """
        Abstract method to generate an agents response to a discussion.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.

        Returns:
        dict[str, str]: A dictionary, with the keys "agreement", "message", and "solution".
        """
