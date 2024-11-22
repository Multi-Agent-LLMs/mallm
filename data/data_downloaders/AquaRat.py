import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class MMLUProDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="aqua_rat",
            dataset_name="deepmind/aqua_rat",
            version="raw",
            sample_size=sample_size,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []

        for sample in data.iter(batch_size=1):
            answers = sample["options"][0]
            formatted_answers = [answer.replace(")", ") ", 1) for answer in answers]
            correct_answer = sample["correct"][0]
            correct_answer_text = next(answer for answer in formatted_answers if answer.startswith(correct_answer))

            question_text = self._clean_text(sample["question"][0])
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[question_and_answers],
                    context=None,
                    references=[correct_answer_text],
                )
            )
        return input_examples