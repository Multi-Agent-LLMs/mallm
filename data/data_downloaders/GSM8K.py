import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class GSM8KDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(name="gsm8k", sample_size=sample_size, trust_remote_code=trust_remote_code)

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[s["question"][0]],
                    context=None,
                    references=[s["answer"][0]],
                )
            )
        return input_examples
