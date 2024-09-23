import logging

from mallm.models.Chat import Chat
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class CriticalResponseGenerator(FreeTextResponseGenerator):
    """
    The SimpleResponseGenerator class is designed to generate simple, straightforward responses based on the input prompts and tasks.
    Its prompts are simplified to mitigate bloating the context length and not distacting the LLM from what is important.
    """

    _name = "critical"

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
            "content": "Critically evaluate the current solution. Identify potential weaknesses or areas of improvement. If you believe the solution is flawless, answer with [AGREE], otherwise answer with [DISAGREE] and provide constructive feedback with suggestions for improvement.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Critically evaluate the task and propose a solution.",
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
            "content": "Improve the current solution. Identify specific areas that need enhancement and propose unique solutions based on your persona. If you see no room for improvement, answer with [AGREE], otherwise, answer with [DISAGREE] and provide a clear, solution.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Propose a well-thought-out unique solution based on your persona to the task.",
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
            "content": "Re-examine the current solution critically based on the feedback provided. Ensure your revision addresses any identified weaknesses or areas for improvement. Submit a revised and improved solution.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "Propose a solution to the task.",
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
        prompt_str = f"""You are participating in a discussion aimed at solving a task.
Task: {data.task_instruction}
Input: {data.input_str}
Your role: {data.persona} ({data.persona_description})
Current Solution: {data.current_draft}
"""  # input has context appended

        appendix = ""
        if data.current_draft is None:
            appendix += "\nNo solution has been proposed yet. Provide a solution that demonstrates critical thinking."
        if data.agent_memory is not None and data.agent_memory != []:
            appendix += "\nThis is the discussion up to this point: \n"
        prompt = [
            {
                "role": "system",
                "content": prompt_str + appendix,
            }
        ]
        if data.agent_memory is not None and data.agent_memory != []:
            prompt += data.agent_memory
        return prompt
