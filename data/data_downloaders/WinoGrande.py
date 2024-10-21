import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample
import random


class WinoGrandeDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False,
    ):
        super().__init__(
            name="winogrande",
            version="winogrande_s",
            dataset_name="allenai/winogrande",
            sample_size=sample_size,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("validation", False)
        input_examples = []
        for sample in data.iter(batch_size=1):
            multiple_choices = [sample["option1"][0], sample["option2"][0]]
            question_text = self._clean_text(sample["sentence"][0])
            formatted_answers = self._format_answer_choices(multiple_choices)
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)
            if sample["answer"][0] == '1':
                ref = formatted_answers[0]
            else:
                ref = formatted_answers[1]

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[question_and_answers],
                    context=None,
                    references=[ref],
                )
            )
        input_examples = input_examples[: self.sample_size]
        return input_examples
