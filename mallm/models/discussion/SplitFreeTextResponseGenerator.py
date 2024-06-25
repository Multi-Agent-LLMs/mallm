import logging
from typing import Optional

from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class SplitFreeTextResponseGenerator(FreeTextResponseGenerator):

    _name = "splitfreetext"

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        prefix = ""
        agreement: Optional[bool] = False
        if data.current_draft:
            agreement = self.generate_agreement(data)
            prefix = {
                None: "",
                True: "You agree with the current solution. ",
                False: "You disagree with the current solution. ",
            }[agreement]

        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": f"{prefix}Based on the current solution, give constructive feedback.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, agreement, False, False
        )

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        prefix = ""
        agreement: Optional[bool] = False
        if data.current_draft:
            agreement = self.generate_agreement(data)
            prefix = {
                None: "",
                True: "You agree with the current solution. ",
                False: "You disagree with the current solution. ",
            }[agreement]

        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": f"{prefix}Improve the current solution.",
            },
        ]
        return self.generate_response(
            current_prompt, chain_of_thought, agreement, False, False
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
