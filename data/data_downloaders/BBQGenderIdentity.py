import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class BBQGenderIdentityDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="bbq_gender",
            dataset_name="heegyu/bbq",
            version="Gender_identity",
            sample_size=sample_size,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []

        for sample in data.iter(batch_size=1):
            question_text = self._clean_text(sample["question"][0])
            formatted_answers = self._format_answer_choices([sample["ans0"][0], sample["ans1"][0], sample["ans2"][0]])
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)

            metadata = {
                "questionIndex": sample["question_index"][0],
                "questionPolarity": sample["question_polarity"][0],
                "contextCondition": sample["context_condition"][0],
                "category": sample["category"][0],
                "additional": sample["additional_metadata"][0]
            }

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=str(sample["example_id"][0]),
                    inputs=[question_and_answers],
                    context=[sample["context"][0]],
                    references=[formatted_answers[sample["label"][0]]],
                    metadata=metadata,
                )
            )
        return input_examples