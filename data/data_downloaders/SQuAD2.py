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
            name="squad_v2",
            version="squad_v2",
            dataset_name="rajpurkar/squad_v2",
            trust_remote_code=trust_remote_code,
            sample_size=sample_size
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
                )
            )
        return input_examples
