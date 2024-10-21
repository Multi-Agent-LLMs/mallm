import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="europarl",
            version="de-en",
            dataset_name="Helsinki-NLP/europarl",
            sample_size=sample_size,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[s["translation"][0]["de"]],
                    context=None,
                    references=[s["translation"][0]["en"]],
                )
            )
        return input_examples
