import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class MathLvl5Downloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="math-lvl-5",
            dataset_name="AI-MO/aimo-validation-math-level-5",
            sample_size=sample_size,
            version="default",
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=str(s["id"][0]),
                    inputs=[s["problem"][0]],
                    context=None,
                    references=[s["answer"][0]],
                )
            )
        return input_examples
