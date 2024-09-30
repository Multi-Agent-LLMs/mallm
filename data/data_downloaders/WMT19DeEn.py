import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class WMT19Downloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="wmt19_de_en",
            version="de-en",
            dataset_name="wmt/wmt19",
            sample_size=sample_size,
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("validation")
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
