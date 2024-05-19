import json
import random
import urllib
import uuid

from data.data_download import DatasetDownloader


class StrategyGADownloader(DatasetDownloader):
    def custom_download(self):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/strategyqa/task.json",
            self.dataset_path,
        )
        self.dataset = json.loads(open(self.dataset_path, encoding="utf-8").read())[
            "examples"
        ]
        random.shuffle(self.dataset)

    def __init__(self):
        super().__init__("strategyqa", hf_dataset=False)

    def process_data(self):
        json_str = ""
        for s in self.dataset[: self.sample_size]:
            ref = [k for k, v in s["target_scores"].items() if v == 1]
            multiple_choice_str = " Answer Choices:"
            for i, (k, v) in enumerate(s["target_scores"].items()):
                multiple_choice_str += " " + f"{chr(ord('A') + i)}) " + k
            json_str += f"""{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": null, "input":[{json.dumps(s['input'] + multiple_choice_str)}], "context": null, "references": [{json.dumps(s['target'])}], "personas": null }}\n"""
        self.save_to_json(json_str)
