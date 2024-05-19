import json
import uuid

from data.data_download import DatasetDownloader


class ETPCDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(name="etpc", version="default", dataset_name="jpwahle/etpc")

    def process_data(self):
        data = self.shuffle_and_select("train")
        json_str = ""
        for s in data.iter(batch_size=1):
            if s["etpc_label"][0] == 1:
                paraphrase_types_str = "Paraphrase Types: "
                for p in list(set(s["paraphrase_types"][0])):
                    paraphrase_types_str += p + ", "
                paraphrase_types_str = paraphrase_types_str[:-2]
                json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": "{s["idx"][0]}", "input":[{json.dumps(s['sentence1'][0])}], "context": [{json.dumps(paraphrase_types_str)}], "references": [{json.dumps(s['sentence2'][0])}], "personas": null }}\n"""
        self.save_to_json(json_str)
