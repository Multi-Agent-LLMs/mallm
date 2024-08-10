import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class XSUMDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="xsum",
            version="xsum",
            dataset_name="GEM/xsum",
            trust_remote_code=True,
            sample_size=sample_size,
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("test")
        input_examples = []
        for s in data.iter(batch_size=1):
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=s["xsum_id"][0],
                    inputs=[s["document"][0]],
                    context=None,
                    references=s["references"][0],
                )
            )
        return input_examples
