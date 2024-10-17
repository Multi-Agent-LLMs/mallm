from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from mallm.models.Chat import Chat
from mallm.utils.types import Response, TemplateFilling

if TYPE_CHECKING:
    from mallm.agents.panelist import Panelist


class ResponseGenerator(ABC):
    @abstractmethod
    def __init__(self, llm: Chat):
        pass

    @abstractmethod
    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        task_instruction: str,
        input_str: str,
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
        task_instruction (str): The instruction for the task and appended context.
        input_str (str): The input for the task.
        chain_of_thought (bool): Whether to use Zero-Shot CoT.

        Returns:
        Response: An object with the attributes "agreement", "message", and "solution".
        """

    @abstractmethod
    def generate_ablation(
        self,
        task_instruction: str,
        input_str: str,
        current_solution: str,
        chain_of_thought: bool,
    ) -> Response:
        """
        Abstract method to generate the response of a single LM that iteratively improves on the current solution.

        Parameters:
        task_instruction (str): The instruction for the task and appended context.
        input_str (str): The input for the task.
        current_solution: (str): The current solution to improve.
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
    def generate_final_answer_prompt(
        input_sample: str,
        task: str,
        previous_answer: Optional[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are tasked with creating a final solution based on the given input and your previous response.",
            },
            {
                "role": "user",
                "content": f"Task: {task}\nInput: {input_sample}\nYour previous response: {previous_answer}",
            },
            {
                "role": "user",
                "content": "Extract the final solution to the task from the provided text. Remove statements of agreement, disagreement, and explanations. Do not modify the text. Do not output any text besides the solution. If there is no solution provided, just copy the previous response.",
            },
        ]

    @staticmethod
    def generate_answer_confidence_prompt(
        panelist: Panelist,
        question: str,
        task: str,
        final_answer: str,
    ) -> list[dict[str, str]]:
        prompts = [
            {
                "role": "system",
                "content": f"Your role: {panelist.persona} ({panelist.persona_description})",
            }
        ]
        discussion_history = panelist.get_discussion_history()[0]
        if discussion_history:
            prompts.append(
                {
                    "role": "user",
                    "content": "Here is the discussion history to help you make a decision:",
                }
            )
            prompts.extend(discussion_history)
        prompts.append(
            {
                "role": "user",
                "content": f"The task is: {task}. The question is: {question}. This is the final answer you provided: '{final_answer}'. Based on this information, please generate a confidence score between 0 and 100 for the final answer. Be critical and only answer with the number.",
            }
        )
        return prompts

    @staticmethod
    def voting_base_prompt(
        voting_message: str,
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = [
            {
                "role": "system",
                "content": f"Your role: {panelist.persona} ({panelist.persona_description})",
            }
        ]
        if history:
            discussion_history = panelist.get_discussion_history()[0]
            if discussion_history:
                prompts.append(
                    {
                        "role": "user",
                        "content": "Here is the discussion history to help you make a decision:",
                    }
                )
                prompts.extend(discussion_history)
        additional_context_str = (
            f"\nAdditional Context: {additional_context}" if additional_context else ""
        )
        content_str = (
            f"{voting_message}\n"
            f"Task: {task}\n"
            f"Question: {question}"
            f"{additional_context_str}\n\n"
            "Here are the possible solutions:"
        )
        prompts.append(
            {
                "role": "user",
                "content": content_str,
            }
        )
        for i, solution in enumerate(solutions):
            confidence_str = (
                ""
                if confidence is None
                else f"\n\n(Confidence: {round(confidence[i])} %)"
            )
            prompts.append(
                {
                    "role": "user",
                    "content": f"Solution {i if anonymous else panelists[i].persona}: {solution}{confidence_str}",
                }
            )
        return prompts

    @staticmethod
    def generate_voting_prompt(
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = ResponseGenerator.voting_base_prompt(
            "You are tasked with voting for the best solution from the list provided below based on the given task.",
            panelist,
            panelists,
            task,
            question,
            solutions,
            additional_context,
            anonymous,
            confidence,
            history,
        )

        prompts.append(
            {
                "role": "user",
                "content": "Based on the above solutions, please provide the number of the solution you are voting for. Answer only with the number.",
            }
        )

        return prompts

    @staticmethod
    def generate_approval_voting_prompt(
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = ResponseGenerator.voting_base_prompt(
            "You are tasked with approving any number of solutions from the list provided below based on the given task.",
            panelist,
            panelists,
            task,
            question,
            solutions,
            additional_context,
            anonymous,
            confidence,
            history,
        )

        prompts.append(
            {
                "role": "user",
                "content": "Based on the above solutions, please provide the numbers of the solutions you are approving, separated by commas. Answer only with the numbers.",
            }
        )

        return prompts

    @staticmethod
    def generate_cumulative_voting_prompt(
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = ResponseGenerator.voting_base_prompt(
            "You are tasked with distributing 10 points among the provided solutions based on the given task.",
            panelist,
            panelists,
            task,
            question,
            solutions,
            additional_context,
            anonymous,
            confidence,
            history,
        )

        prompts.append(
            {
                "role": "user",
                "content": "Based on the above solutions, please distribute 10 points among the solutions. Provide your points allocation as a JSON dictionary where keys are solution numbers (as int) and values are the points. The total points should sum up to 10. Answer only with the JSON dictionary.",
            }
        )

        return prompts

    @staticmethod
    def generate_ranking_prompt(
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = ResponseGenerator.voting_base_prompt(
            "You are tasked with ranking the solutions from the most preferred to the least preferred based on the given task.",
            panelist,
            panelists,
            task,
            question,
            solutions,
            additional_context,
            anonymous,
            confidence,
            history,
        )

        prompts.append(
            {
                "role": "user",
                "content": "Based on the above solutions, please provide the rankings of the solutions separated by spaces. Example: '0 2 1' if you prefer Solution 0 the most, then Solution 2, and finally Solution 1. Provide up to 5 rankings. Only answer with the rankings.",
            }
        )

        return prompts

    @staticmethod
    def generate_summary_prompt(
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]:
        prompts = []

        # Add discussion history if available
        if history:
            discussion_history = panelist.get_discussion_history()[0]
            if discussion_history:
                prompts.append(
                    {
                        "role": "user",
                        "content": "Here is the discussion history to help you make a decision:",
                    }
                )
                prompts.extend(discussion_history)

        # Prepare the main content for the summary request
        additional_context_str = (
            f"\nAdditional Context: {additional_context}" if additional_context else ""
        )

        content_str = (
            f"Task: {task}\n"
            f"Question: {question}"
            f"{additional_context_str}\n\n"
            "Please provide a summary of the following solutions and combine them in a single answer to solve the task. Only answer with the solution:"
        )

        # Add each solution to the content string
        for i, solution in enumerate(solutions):
            confidence_str = (
                "" if confidence is None else f" (Confidence: {round(confidence[i])}%)"
            )
            panelist_label = f"Solution {i}" if anonymous else f"{panelists[i].persona}"
            content_str += f"\n\n{panelist_label}: {solution}{confidence_str}"

        # Append the final content as a user message
        prompts.append(
            {
                "role": "user",
                "content": content_str,
            }
        )

        # Return the prompts list
        return prompts
