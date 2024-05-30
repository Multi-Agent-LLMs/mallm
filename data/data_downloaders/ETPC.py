import json
import uuid

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class ETPCDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(name="etpc", version="default", dataset_name="jpwahle/etpc")

    def process_data(self) -> list[InputExample]:
        data = self.shuffle_and_select("train")
        input_examples = []
        for s in data.iter(batch_size=1):
            if s["etpc_label"][0] == 1:
                paraphrase_types_str = "Paraphrase Types: "
                for p in list(set(s["paraphrase_types"][0])):
                    paraphrase_types_str += p + ", "
                paraphrase_types_str = paraphrase_types_str[:-2]
                input_examples.append(
                    InputExample(
                        example_id=str(uuid.uuid4()),
                        dataset_id=s["idx"][0],
                        input_str=[s["sentence1"][0]],
                        context=[paraphrase_types_str],
                        references=[s["sentence2"][0]],
                        personas=None,
                    )
                )
        return input_examples
