import dataclasses
import json
import os
from typing import Any, Optional, Union

from mallm.evaluation.metrics.metric import Metric
from mallm.utils.evaluation import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


class IFEval(Metric):
    """
    A class to evaluate generated texts by instruction-following, following https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_main.py
    """

    _name = "IFEval"

    @staticmethod
    def read_prompt_list(input_jsonl_filename: str) -> list[InputExample]:
        """Read inputs from jsonl."""
        inputs = []
        with open(input_jsonl_filename) as f:
            for lel in f:
                example = json.loads(lel)
                inputs.append(
                        InputExample(key=example["key"],
                            instruction_id_list=example["instruction_id_list"],
                            prompt=example["prompt"],
                            kwargs=example["kwargs"]))
        return inputs

    @staticmethod
    def test_instruction_following_strict(
            inp: InputExample,
            prompt_to_response: dict[str, str],
    ) -> OutputExample:
        """Tests response to see if instrutions are followed."""
        response = prompt_to_response[inp.prompt]
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)    # type: ignore[no-untyped-call]

            instruction.build_description(**inp.kwargs[index])   # type: ignore[no-untyped-call]
            args = instruction.get_instruction_args()    # type: ignore[no-untyped-call]
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)     # type: ignore[no-untyped-call]

            if response.strip() and instruction.check_following(response):   # type: ignore[no-untyped-call]
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        return OutputExample(
                instruction_id_list=inp.instruction_id_list,
                prompt=inp.prompt,
                response=response,
                follow_all_instructions=all(is_following_list),
                follow_instruction_list=is_following_list,
        )

    @staticmethod
    def test_instruction_following_loose(
            inp: InputExample,
            prompt_to_response: dict[str, str],
    ) -> OutputExample:
        """Tests response for an upper bound for following instructions."""
        response = prompt_to_response[inp.prompt]
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
                response,
                revised_response,
                response_remove_first,
                response_remove_last,
                response_remove_both,
                revised_response_remove_first,
                revised_response_remove_last,
                revised_response_remove_both,
        ]
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)    # type: ignore[no-untyped-call]

            instruction.build_description(**inp.kwargs[index])   # type: ignore[no-untyped-call]
            args = instruction.get_instruction_args()    # type: ignore[no-untyped-call]
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)     # type: ignore[no-untyped-call]

            is_following = False
            for r in all_responses:     # type: ignore[assignment]
                if r.strip() and instruction.check_following(r):     # type: ignore[no-untyped-call, attr-defined]
                    is_following = True
                    break

            is_following_list.append(is_following)

        return OutputExample(
                instruction_id_list=inp.instruction_id_list,
                prompt=inp.prompt,
                response=response,
                follow_all_instructions=all(is_following_list),
                follow_instruction_list=is_following_list,
        )

    @staticmethod
    def read_prompt_to_response_dict(input_jsonl_filename: str) -> dict[str, str]:
        """Creates dictionary matching prompt and response."""
        return_dict = {}
        with open(input_jsonl_filename) as f:
            for lel in f:
                example = json.loads(lel)
                return_dict[example["prompt"]] = example["response"]
        return return_dict

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str], dataset_id: Optional[str]) -> dict[str, Any]:

        inputs = IFEval().read_prompt_list(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) + "/utils/evaluation/input_data.jsonl")
        inputs = [input_obj for input_obj in inputs if str(input_obj.key) == str(dataset_id)]
        prompt_to_response = {}
        prompt_to_response[inputs[0].prompt] = generated_text

        # get instruction following results
        outputs_strict = IFEval().test_instruction_following_strict(inputs[0], prompt_to_response)
        outputs_loose = IFEval().test_instruction_following_loose(inputs[0], prompt_to_response)

        return {"follow_all_instructions_strict": outputs_strict.follow_all_instructions,
                "instruction_id_list_strict": outputs_strict.instruction_id_list,
                "follow_instruction_list_strict": outputs_strict.follow_instruction_list,
                "follow_all_instructions_loose": outputs_loose.follow_all_instructions,
                "instruction_id_list_loose": outputs_loose.instruction_id_list,
                "follow_instruction_list_loose": outputs_loose.follow_instruction_list
                }
