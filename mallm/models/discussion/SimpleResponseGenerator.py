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
            "content": "Improve the current solution. If you agree with the current solution, answer with [AGREE], else answer with [DISAGREE] and explain why and provide an improved solution.",
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
        prompt_str = f"""You take part in a discussion to solve a task.
Task: {data.task_instruction}
Input: {data.input_str}
Your role: {data.persona} ({data.persona_description})"""

        appendix = ""
        if data.current_draft:
            appendix += (
                f"\nCurrent Solution: {data.current_draft}"
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

    def generate_judgement(self, data: TemplateFilling, answer_before: str, answer_after: str) -> Response:

        prompt_template = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
        Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
        Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

        [User Question]
        {input}

        [The Start of Assistant A's Answer]
        {response_a}
        [The End of Assistant A's Answer]

        [The Start of Assistant B's Answer]
        {response_b}
        [The End of Assistant B's Answer]
        """

        current_prompt = [{"role": "user", "content": prompt_template.format(input=data.task_instruction, response_a=answer_before, response_b=answer_after)}]

        return self.generate_response(
            current_prompt,
            data.task_instruction,
            data.input_str,
            False,
            None,
            False,
            False,
        )

    def generate_policy_intervention(self, data: TemplateFilling, provide_labels: bool = True) -> Response:
        instr_prompt = {
            "role": "user",
            "content": "The currect discussion is going badly. Based on the others contributions, give constructive feedback about how to improve the discussion habits. Be concise so that the other discussion participants can find a better solution.",
        }
        if provide_labels:
            error_categories = [
                "Task Compliance: Off-topic, Bad instruction following",
                "Lack of Progress: Inefficiency, Redundancy, Circular discussion, Repetition, Unproductive disagreement",
                "Low Quality Engagement: Poor collaboration, Minimal participation, Disjointed contribution, Ignorance",
                "Low-Quality Feedback: Excessive criticism, Excessive agreement, Self-contradictory feedback, Unhelpful feedback",
                "Lack of Clarity: Overanalysis, Overgeneralization, Insignificant changes",
                "Knowledge Gap: Assumptions, Lack of data, Hallucinated facts, Wrongly cited",
                "Logical Errors: Lack of common sense, Reasoning error",
                "Linguistic Errors: Fluency, Grammatical errors, False pronouns",
                "Other: Describe as explanation"
            ]
            instr_prompt["content"] += f"\nThe following problematic error categories exist. If you identify them in the current discussion, they could help you to provide better feedback:\n {error_categories}"

        current_prompt = [
            *self.get_filled_template(data),
            instr_prompt,
        ]
        return self.generate_response(
            current_prompt,
            data.task_instruction,
            data.input_str,
            False,
            None,
            False,
            False,
        )
