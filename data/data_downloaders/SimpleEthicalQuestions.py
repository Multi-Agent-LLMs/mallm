import json
import os
import random
import urllib
import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class SimpleEthicalQuestionsDownloader(DatasetDownloader):
    def custom_download(self):
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        file_path = os.path.join(self.dataset_path, "task.json")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks"
            "/simple_ethical_questions/task.json",
            file_path,
        )
        self.dataset = json.loads(open(file_path, encoding="utf-8").read())["examples"]
        random.shuffle(self.dataset)

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="simple_ethical_questions", hf_dataset=False, sample_size=sample_size
        )

    def process_data(self) -> list[InputExample]:
        input_examples = []
        for s in self.dataset[: self.sample_size]:
            ref = [k for k, v in s["target_scores"].items() if v == 1]
            multiple_choice_str = " Answer Choices:"
            for i, (k, v) in enumerate(s["target_scores"].items()):
                multiple_choice_str += " " + f"{chr(ord('A') + i)}) " + k
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[s["input"]],
                    context=[multiple_choice_str],
                    references=[ref[0]],
                    personas=None,
                )
            )
        return input_examples
