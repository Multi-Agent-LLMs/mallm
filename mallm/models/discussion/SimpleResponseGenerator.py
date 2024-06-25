import logging

from mallm.models.Chat import Chat
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class SimpleResponseGenerator(FreeTextResponseGenerator):

    _name = "simple"

    def __init__(self, llm: Chat):
        self.llm = llm

    def generate_baseline(
        self, task_instruction: str, input_str: str, chain_of_thought: bool
    ) -> Response:
        prompt_content = f"""
{task_instruction}
Input: {input_str}
"""  # input has context appended
        prompt = [
            {
                "role": "system",
                "content": prompt_content,
            },
        ]
        return self.generate_response(prompt, chain_of_thought, None, True, True)

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Based on the current solution, give constructive feedback. If you agree, answer with [AGREE], else answer with [DISAGREE] and explain why.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, False
        )

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Improve the current solution. If you agree with the current solution, answer with [AGREE], else answer with [DISAGREE] and explain why and provide an improved solution.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, False
        )

    def generate_draft(self, data: TemplateFilling, chain_of_thought: bool) -> Response:
        current_prompt = [
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Based on the provided feedback, carefully re-examine your previous solution. Provide a revised solution based on the feedback.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, None, False, True
        )

    @staticmethod
    def get_filled_template(data: TemplateFilling) -> list[dict[str, str]]:
        prompt_str = f"""You take part in a discussion to solve a task.
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
                "role": "system",
                "content": prompt_str + appendix,
            }
        ]
        if data.agent_memory is not None and data.agent_memory != []:
            prompt += data.agent_memory
        return prompt
