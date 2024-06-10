import json
import uuid

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="squad_v2",
            version="squad_v2",
            dataset_name="rajpurkar/squad_v2",
            trust_remote_code=True,
            sample_size=1000,
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=s["id"][0],
                    inputs=[s["question"][0]],
                    context=[s["context"][0]],
                    references=s["answers"][0]["text"],
                    personas=None,
                )
            )
        return input_examples
