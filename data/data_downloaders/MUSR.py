import ast
import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class MUSRDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="musr",
            dataset_name="TAUR-Lab/MuSR",
            version="default",
            sample_size=sample_size,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("murder_mysteries")
        input_examples = []

        for sample in data.iter(batch_size=1):
            answers = ast.literal_eval(sample["choices"][0])
            correct_answer = sample["answer_choice"][0]

            question_text = self._clean_text(sample["question"][0])
            context_text = self._clean_text(sample["narrative"][0])
            question_and_answers = f"{question_text} {' or '.join(answers)}?"

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[question_and_answers],
                    context=[context_text],
                    references=[correct_answer],
                )
            )
        return input_examples

    @staticmethod
    def _clean_text(text):
        return (
            text.replace("\n", " ")
            .replace("\r", " ")
            .replace('"', "")
            .replace("\\n", " ")
        )
