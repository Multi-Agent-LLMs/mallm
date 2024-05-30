import json
import uuid

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class GSM8KDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__("gsm8k")

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    input_str=[s["question"][0]],
                    context=None,
                    references=[s["answer"][0]],
                    personas=None,
                )
            )
        return input_examples
