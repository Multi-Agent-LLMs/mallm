import json
import uuid

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="europarl",
            version="de-en",
            dataset_name="Helsinki-NLP/europarl",
            sample_size=1000,
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
                    personas=None,
                )
            )
        return input_examples
