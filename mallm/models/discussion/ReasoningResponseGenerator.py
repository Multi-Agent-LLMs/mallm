import logging

from mallm.models.Chat import Chat
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class ReasoningResponseGenerator(FreeTextResponseGenerator):
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
                "content": "You are starting the discussion. Lay out your thoughts and ideas but don't provide a final solution yet.",
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
            "content": "Improve the current steps of the argument by referring to the other participants in the discussion. Be critical and answer short and concise. Repeat only the reasoning steps that you think are the most important. If you think there is enough information to create a final answer also answer with [AGREE] else answer with [DISAGREE]. Don't provide a final solution yet.",
        }
        if not data.current_draft:
            instr_prompt = {
                "role": "user",
                "content": "You are starting the discussion. Lay out your thoughts and ideas but don't provide a final solution yet.",
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
                "content": "You are starting the discussion. Lay out your thoughts and ideas but don't provide a final solution yet.",
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
        prompt_str = f"""You take part in a discussion to solve a task. Don't answer with too much information at once. Be concise and clear.
Task: {data.task_instruction}
Question: {data.input_str}
Your role: {data.persona} ({data.persona_description})
"""

        appendix = ""
        if data.agent_memory is not None and data.agent_memory != []:
            appendix += "\nThis is the discussion to the current point: \n"
        prompt = [
            {
                "role": "system",
                "content": prompt_str + appendix,
            }
        ]
        if data.agent_memory is not None and data.agent_memory != []:
            prompt += data.agent_memory
        return prompt

    @staticmethod
    def generate_final_answer_prompt(
        persona: str,
        persona_description: str,
        question: str,
        task: str,
        previous_answer: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"You are {persona}, {persona_description}. Use your unique perspective to provide a concise solution.",
            },
            {
                "role": "user",
                "content": (
                    f"As {persona}, you are tasked with creating a final solution based on the given question and your previous response.\n\n"
                    f"Task: {task}\n"
                    f"Question: {question}\n"
                    f"Your previous solution: {previous_answer}\n\n"
                    f"As {persona}, please provide a concise final solution to the task based on your unique insights. "
                    "Ensure your answer is original and directly addresses the question without additional explanations."
                ),
            },
        ]
