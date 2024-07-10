from abc import ABC, abstractmethod
from typing import Optional

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
        agreement: Optional[bool],
        baseline: bool,
        drafting: bool,
    ) -> Response:
        """
        Abstract method to generate an agents response to a discussion.

        Parameters:
        current_prompt (list[dict[str, str]]): The prompt for the LM.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.
        agreement (Optional[bool]): the agreement if already computed.
        baseline (bool): Whether use the prompt for the baseline, without discussion.
        drafting (bool): Whether the response should be drafting a new solution.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
        """

    @abstractmethod
    def generate_baseline(
        self, task_instruction: str, input_str: str, chain_of_thought: bool
    ) -> Response:
        """
        Abstract method to generate the response of a single LM as a baseline.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.
        task_instruction (str): The instruction for the task and appended context.
        input_str (str): The input for the task.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
        """

    @abstractmethod
    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        """
        Abstract method to generate feedback to improve a draft for a discussion.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
        """

    @abstractmethod
    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        """
        Abstract method to generate feedback and an improved solution for a discussion.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
        """

    @abstractmethod
    def generate_draft(self, data: TemplateFilling, chain_of_thought: bool) -> Response:
        """
        Abstract method to generate a draft for a discussion.

        Parameters:
        data (TemplateFilling): The fields used for prompting the LM.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
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

    @staticmethod
    def get_filled_template_slim(data: TemplateFilling) -> list[dict[str, str]]:
        prompt_str = f"""
Task: {data.task_instruction}
Input: {data.input_str}
Your role: {data.persona} ({data.persona_description})
Current Solution: {data.current_draft}
"""  # input has context appended

        appendix = ""
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

    @staticmethod
    def extract_agreement(res: str, drafting: bool) -> Optional[bool]:
        """
        Determines whether a string given by an agent means an agreement or disagreement.
        Returns bool
        """
        if drafting:
            return None
        return "agree" in res.lower() and "disagree" not in res.lower()

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
