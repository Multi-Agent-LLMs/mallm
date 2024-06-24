from abc import ABC, abstractmethod

from mallm.models.Chat import Chat
from mallm.utils.types import Response, TemplateFilling


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
    ) -> Response:
        """
        Abstract method to generate an agents response to a discussion.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.

        Returns:
        dict[str, str]: A dictionary, with the keys "agreement", "message", and "solution".
        """

    @staticmethod
    def get_filled_template(data: TemplateFilling) -> list[dict[str, str]]:
        prompt_str = f"""
Task: {data.task_instruction}
Input: {data.input_str}
Your role: {data.persona} ({data.persona_description})
Current Solution: {data.current_draft}
"""  # input has context appended

        appendix = ""
        if data.feedback_sentences is not None:
            appendix += f"\nExplain your feedback and solution in {data.feedback_sentences[0]} to {data.feedback_sentences[1]} sentences!"
        if data.current_draft is None:
            appendix += (
                "\nNobody proposed a solution yet. Please provide the first one."
            )
        if data.agent_memory is not None and data.agent_memory != []:
            appendix += "\nThis is the discussion to the current point: \n"
        prompt = [
            {
                "role": "user",
                "content": prompt_str + appendix,
            }
        ]
        if data.agent_memory is not None and data.agent_memory != []:
            prompt += data.agent_memory
        return prompt
