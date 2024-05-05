import json
import uuid

from data.data_download import DatasetDownloader


class GSM8KDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__("gsm8k")

    def process_data(self):
        data = self.shuffle_and_select("test")
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": null, "input":{json.dumps(s['question'][0])}, "context": null, "references": [{json.dumps(s['answer'][0])}], "personas": null }}\n"""
        self.save_to_json(json_str)
