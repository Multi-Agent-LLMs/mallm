import json
import logging
from typing import Optional

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class FreeTextResponseGenerator(ResponseGenerator):

    _name = "freetext"

    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt = {
            "role": "system",
            "content": "You are participating in a discussion to solve the provided task.",
        }

        self.base_prompt_baseline = {
            "role": "system",
            "content": "Solve the provided task.",
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
        return self.generate_response(prompt, chain_of_thought, None, True, True)

    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        chain_of_thought: bool,
        agreement: Optional[bool],
        baseline: bool,
        drafting: bool,
    ) -> Response:
        if chain_of_thought:
            current_prompt.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
                }
            )
        logger.debug(f"Sending prompt: {json.dumps(current_prompt, indent=2)}")

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
                solution=self.extract_result(res),
            )

            if response.agreement is None and not drafting and not baseline:
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
            current_prompt, chain_of_thought, None, False, False
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
            current_prompt, chain_of_thought, None, False, False
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
            current_prompt, chain_of_thought, None, False, True
        )

    def extract_result(self, result: Optional[str]) -> str:
        current_prompt = [
            {
                "role": "system",
                "content": "Extract the final solution to the task from the provided text. Remove statements of agreement, disagreement, and explanations. Do not modify the text. Do not output any text besides the solution. If there is no solution provided, just copy the text.",
            },
            {
                "role": "user",
                "content": f"Text: {result}",
            },
            {
                "role": "assistant",
                "content": "Final solution:",
            },
        ]
        return self.llm.invoke(current_prompt)
