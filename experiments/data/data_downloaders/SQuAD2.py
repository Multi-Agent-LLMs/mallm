import json
import uuid

from experiments.data.data_download import DatasetDownloader


class EuroparlDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(name='squad_v2', version="squad_v2", dataset_name="rajpurkar/squad_v2")

    def process_data(self):
        data = self.shuffle_and_select('train')
        json_str = ""
        for s in data.iter(batch_size=1):
            json_str += f'''{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": "{s["id"][0]}", "input":{json.dumps(s['question'][0])}, "context": {json.dumps(s['context'][0])}, "references": [{json.dumps(s['answers'][0]["text"])}], "personas": null }}\n'''
        self.save_to_json(json_str)
