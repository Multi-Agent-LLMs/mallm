import json
import uuid

from data.data_download import DatasetDownloader


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="multi_news", version="multi_news", dataset_name="multi_news"
        )

    def process_data(self):
        data = self.shuffle_and_select("test")
        examples = []

        for sample in data.iter(batch_size=1):
            docs = sample["document"].split(" ||||| ")
            example = {
                "exampleId": str(uuid.uuid4()),
                "datasetId": None,
                "input": docs,
                "context": None,
                "references": [sample["summary"]],
                "personas": None,
            }
            examples.append(json.dumps(example) + "\n")

        self.save_to_json("".join(examples))
