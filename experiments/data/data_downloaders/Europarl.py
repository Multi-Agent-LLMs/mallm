import json
import uuid

from experiments.data.data_download import DatasetDownloader


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="europarl", version="de-en", dataset_name="Helsinki-NLP/europarl"
        )

    def process_data(self):
        data = self.shuffle_and_select("train")
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": null, "input":{json.dumps(s['translation'][0]['de'])}, "context": null, "references": [{json.dumps(s['translation'][0]['en'])}], "personas": null }}\n"""
        self.save_to_json(json_str)
