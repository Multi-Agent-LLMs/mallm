import logging
from typing import Optional

from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class SplitFreeTextResponseGenerator(FreeTextResponseGenerator):
    """
    The SplitFreeTextResponseGenerator class is designed to generate free-form text responses based on the input prompts and tasks.
    It makes two queries instead of one: 1) generate the agreement 2) generate the corresponding message depending on the agreement
    With this split-up method, the agreement extraction can work reliably.
    """

    _name = "splitfreetext"

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        prefix = ""
        agreement: Optional[bool] = False
        instr_prompt = {
            "role": "user",
            "content": "Give constructive feedback.",
        }
        if data.current_draft:
            agreement = self.generate_agreement(data)
            prefix = {
                None: "",
                True: "You agree with the current solution. ",
                False: "You disagree with the current solution. ",
            }[agreement]
            instr_prompt = {
                "role": "user",
                "content": f"{prefix}Based on the current solution, give constructive feedback.",
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
            agreement,
            False,
            False,
        )

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        prefix = ""
        agreement: Optional[bool] = False
        instr_prompt = {
            "role": "user",
            "content": "Propose a solution.",
        }
        if data.current_draft:
            agreement = self.generate_agreement(data)
            prefix = {
                None: "",
                True: "You agree with the current solution. ",
                False: "You disagree with the current solution. ",
            }[agreement]
            instr_prompt = {
                "role": "user",
                "content": f"{prefix}Improve the current solution.",
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
            agreement,
            False,
            False,
        )

    def generate_agreement(self, data: TemplateFilling) -> Optional[bool]:
        prompt = [
            self.base_prompt,
            *self.get_filled_template_slim(data),
            {
                "role": "user",
                "content": "Do you agree with the solution, considering the arguments and evidence presented? Please provide your reasoning step-by-step. After that, respond with [AGREE] or [DISAGREE].",
            },
        ]
        return self.extract_agreement(res=self.llm.invoke(prompt), drafting=False)

    def generate_policy_intervention(self, data: TemplateFilling, provide_labels: bool = True) -> Response:
        logger.error(f"Policy Intervention is not implemented for this response generator. {self.__class__.__name__}")
        raise NotImplementedError("Policy Intervention is not implemented for this response generator.")
