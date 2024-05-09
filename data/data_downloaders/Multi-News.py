import json
import uuid

from data.data_download import DatasetDownloader


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="multi_news", version="default", dataset_name="multi_news"
        )

    def process_data(self):
        data = self.shuffle_and_select("test")
        examples = []

        for sample in data.iter(batch_size=1):
            docs = sample["document"][0].split(" ||||| ")
            example = {
                "exampleId": str(uuid.uuid4()),
                "datasetId": None,
                "input": docs,  # note: there exist few samples with just one input doc
                "context": None,
                "references": sample["summary"],
                "personas": None,
            }
            examples.append(json.dumps(example) + "\n")

        self.save_to_json("".join(examples))
