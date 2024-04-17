import json
import uuid

from experiments.data.data_download import DatasetDownloader


class ETPCDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(name="etpc", version="default", dataset_name="jpwahle/etpc")

    def process_data(self):
        data = self.shuffle_and_select("train")
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": "{s["idx"][0]}", "input":{json.dumps(s['sentence1'][0])}, "context": null, "references": [{json.dumps(s['sentence2'][0])}], "personas": null }}\n"""
        self.save_to_json(json_str)
