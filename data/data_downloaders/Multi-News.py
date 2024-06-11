import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class MultiNewsDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="multi_news",
            version="default",
            dataset_name="multi_news",
            trust_remote_code=True,
            sample_size=sample_size,
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []

        for sample in data.iter(batch_size=1):
            docs = sample["document"][0].split(" ||||| ")
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=docs,
                    context=None,
                    references=sample["summary"],
                    personas=None,
                )
            )
        return input_examples
