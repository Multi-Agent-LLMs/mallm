import json
import uuid

from data.data_download import DatasetDownloader


class WMT19Downloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(name="wmt19_de_en", version="de-en", dataset_name="wmt/wmt19")

    def process_data(self):
        data = self.shuffle_and_select("validation")
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": null, "input":[{json.dumps(s['translation'][0]['de'])}], "context": null, "references": [{json.dumps(s['translation'][0]['en'])}], "personas": null }}\n"""
        self.save_to_json(json_str)
