import logging
from typing import Optional

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class FreeTextResponseGenerator(ResponseGenerator):
    """
    The FreeTextResponseGenerator class is designed to generate free-form text responses based on the input prompts and tasks.
    It utilizes a chat interface to interact with the language model and generate responses that are not limited by specific templates or structures.
    """

    _name = "freetext"

    def __init__(self, llm: Chat, judge_llm: Optional[Chat] = None):
        self.llm = llm
        self.judge_llm = judge_llm
        self.base_prompt = {
            "role": "system",
            "content": "You are participating in a discussion to solve the provided task.",
        }
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
                "role": "user",
                "content": prompt_content,
            },
        ]
        return self.generate_response(
            prompt, task_instruction, input_str, chain_of_thought, None, True, True
        )

    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        task_instruction: str,
        input_str: str,
        chain_of_thought: bool,
        agreement: Optional[bool],
        baseline: bool,
        drafting: bool,
        judging: bool = False,
    ) -> Response:
        if chain_of_thought:
            current_prompt.append(
                {
                    "role": "user",
                    "content": "Let's think step by step.",
                }
            )

        retry = 0
        while retry < 10:
            res = self.llm.invoke(current_prompt)

            response = Response(
                agreement=(
                    agreement
                    if agreement is not None
                    else self.extract_agreement(res, drafting)
                ),
                message=res,
                solution=self.extract_result(res, task_instruction, input_str),
            )

            if response.agreement is None and not drafting and not baseline and not judging:
                retry += 1
                continue
            break  # success
        if retry >= 10:
            logger.error(
                f"After 10 retries the response could not be decoded. \nPrompt: {current_prompt} \nResponse string: {response}"
            )
            raise Exception("Could not decode response.")
        return response

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
            self.base_prompt,
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
            self.base_prompt,
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
            self.base_prompt,
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

    def extract_result(
        self, result: Optional[str], task_instruction: str, input_text: str
    ) -> str:
        current_prompt = self.generate_final_answer_prompt(
            input_sample=input_text,
            task=task_instruction,
            previous_answer=result
        )
        return str(self.llm.invoke(current_prompt))

    def generate_ablation(
        self,
        task_instruction: str,
        input_str: str,
        current_solution: str,
        chain_of_thought: bool,
    ) -> Response:
        prompt_content = f"""
When, faced with a task, improve the current solution.
Task: {task_instruction}
Input: {input_str}
Current solution: {current_solution}
"""  # input has context appended
        prompt = [
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        return self.generate_response(
            current_prompt=prompt,
            task_instruction=task_instruction,
            input_str=input_str,
            chain_of_thought=chain_of_thought,
            agreement=None,
            baseline=True,
            drafting=True,
        )

    def generate_policy_intervention(self, data: TemplateFilling, provide_labels: bool = True) -> Response:
        logger.error(f"Policy Intervention is not implemented for this response generator. {self.__class__.__name__}")
        raise NotImplementedError("Policy Intervention is not implemented for this response generator.")

    def generate_judgement(self, data: TemplateFilling, answer_before: str, answer_after: str) -> Response:
        logger.error(f"Judgement is not implemented for this response generator. {self.__class__.__name__}")
        raise NotImplementedError("Judgement is not implemented for this response generator.")
