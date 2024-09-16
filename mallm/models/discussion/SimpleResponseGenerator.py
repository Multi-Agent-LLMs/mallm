import logging

from mallm.models.Chat import Chat
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class SimpleResponseGenerator(FreeTextResponseGenerator):
    """
    The SimpleResponseGenerator class is designed to generate simple, straightforward responses based on the input prompts and tasks.
    Its prompts are simplified to mitigate bloating the context length and not distacting the LLM from what is important.
    """

    _name = "simple"

    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt_baseline = {
            "role": "system",
            "content": "Solve the provided task. Do not ask back questions. Clearly indicate your final solution after the text 'Final Solution:'.",
        }

    def generate_baseline(
        self, task_instruction: str, input_str: str, chain_of_thought: bool
    ) -> Response:
        prompt_content = f"""
Task: {task_instruction}
Input: {input_str}
        """  # input has context appended
        prompt = [
            self.base_prompt_baseline,
            {
                "role": "system",
                "content": prompt_content,
            },
        ]
        return self.generate_response(
            prompt, task_instruction, input_str, chain_of_thought, None, True, True
        )

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        instr_prompt = {
            "role": "user",
            "content": "Based on the current solution, give constructive feedback. If you agree, answer with [AGREE], else answer with [DISAGREE] and explain why.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Give constructive feedback.",
            }
        current_prompt = [
            *self.get_filled_template(data),
            instr_prompt,
        ]
        return self.generate_response(
            current_prompt,
            data.task_instruction,
            data.input_str,
            chain_of_thought,
            None,
            False,
            False,
        )

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        instr_prompt = {
            "role": "user",
            "content": "Critically evaluate the current solution. Scrutinize every aspect and look for any flaws, inefficiencies, or areas for improvement. If no improvements can be made, respond with [AGREE] and justify why. If there are any shortcomings, respond with [DISAGREE], explain why, and provide a better solution.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Propose a solution.",
            }
        current_prompt = [
            *self.get_filled_template(data),
            instr_prompt,
        ]
        return self.generate_response(
            current_prompt,
            data.task_instruction,
            data.input_str,
            chain_of_thought,
            None,
            False,
            False,
        )

    def generate_draft(self, data: TemplateFilling, chain_of_thought: bool) -> Response:
        instr_prompt = {
            "role": "user",
            "content": "Based on the provided feedback, carefully re-examine your previous solution. Provide a revised solution.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Propose a solution.",
            }
        current_prompt = [
            *self.get_filled_template(data),
            instr_prompt,
        ]
        return self.generate_response(
            current_prompt,
            data.task_instruction,
            data.input_str,
            chain_of_thought,
            None,
            False,
            True,
        )

    @staticmethod
    def get_filled_template(data: TemplateFilling) -> list[dict[str, str]]:
        prompt_str = f"""Task: {data.task_instruction}
Question: {data.input_str}
"""

        appendix = ""
        if data.current_draft is None:
            appendix += (
                "\nNobody proposed a solution yet. Please provide the first one."
            )
        if data.agent_memory is not None and data.agent_memory != []:
            appendix += "\nThis is the discussion to the current point: \n"
        prompt = [
            {
                "role": "system",
                "content": f"You are a participant in a discussion to solve a task. Your role is to provide a solution. Do not ask back questions. You are a {data.persona} more specifically {data.persona_description}.",
            },
            {
                "role": "user",
                "content": prompt_str + appendix,
            },
        ]
        if data.agent_memory is not None and data.agent_memory != []:
            prompt += data.agent_memory
        return prompt
