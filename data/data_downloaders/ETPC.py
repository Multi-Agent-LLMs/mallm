import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class ETPCDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None, trust_remote_code: bool = False
    ):
        super().__init__(
            name="etpc",
            version="default",
            dataset_name="jpwahle/etpc",
            sample_size=sample_size,
            trust_remote_code=trust_remote_code
        )

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train", False)
        input_examples = []
        for s in data.iter(batch_size=1):
            if s["etpc_label"][0] == 1 and s["sentence2"][0]:
                paraphrase_types_str = "Paraphrase Types: "
                for p in list(set(s["paraphrase_types"][0])):
                    paraphrase_types_str += p + ", "
                paraphrase_types_str = paraphrase_types_str[:-2]
                input_examples.append(
                    InputExample(
                        example_id=str(uuid.uuid4()),
                        dataset_id=s["idx"][0],
                        inputs=[s["sentence1"][0]],
                        context=[paraphrase_types_str],
                        references=[s["sentence2"][0]],
                    )
                )
        input_examples = input_examples[: self.sample_size]
        return input_examples
