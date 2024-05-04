import json
import uuid

from data.data_download import DatasetDownloader


class XSUMDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="xsum", version="xsum", dataset_name="GEM/xsum", trust_remote_code=True
        )

    def process_data(self):
        data = self.shuffle_and_select("test")
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": "{s["xsum_id"][0]}", "input":{json.dumps(s['document'][0])}, "context": null, "references": [{json.dumps(s['references'][0])}], "personas": null }}\n"""
        self.save_to_json(json_str)
