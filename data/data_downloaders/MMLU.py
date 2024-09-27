import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class MMLUDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="mmlu",
            dataset_name="cais/mmlu",
            version="all",
            sample_size=sample_size,
            hf_token=hf_token,
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []

        for sample in data.iter(batch_size=1):
            answers = sample["choices"][0]
            correct_answer = chr(65 + sample["answer"][0])

            question_text = self._clean_text(sample["question"][0])
            formatted_answers = self._format_answer_choices(answers)
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[question_and_answers],
                    context=None,
                    references=[correct_answer],
                )
            )
        return input_examples
