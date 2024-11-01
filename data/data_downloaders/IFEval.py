import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class IFEvalDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="ifeval",
            dataset_name="google/IFEval",
            version="default",
            sample_size=sample_size,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train")
        input_examples = []

        for sample in data.iter(batch_size=1):
            question_text = self._clean_text(sample["prompt"][0])

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=str(sample["key"][0]),
                    inputs=[question_text],
                    context=None,
                    references=[],    # Evaluation requires https://github.com/google-research/google-research/tree/master/instruction_following_eval
                )
            )
        return input_examples
