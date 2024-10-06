import json
import os
import random
import urllib
import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


class StrategyGADownloader(DatasetDownloader):
    def custom_download(self):
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        file_path = self.dataset_path + "/task.json"
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/strategyqa/task.json",
            file_path,
        )
        self.dataset = json.loads(open(file_path, encoding="utf-8").read())["examples"]
        random.shuffle(self.dataset)

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(name="strategyqa", hf_dataset=False, sample_size=sample_size)

    def process_data(self) -> list[InputExample]:
        input_examples = []
        for sample in self.dataset[: self.sample_size]:
            ref = [f"{chr(65 + i)}) " + k for i, (k, v) in enumerate(sample["target_scores"].items()) if v == 1]
            multiple_choices = []
            for i in sample["target_scores"]:
                multiple_choices.append(i)
            
            question_text = self._clean_text(sample["input"])
            formatted_answers = self._format_answer_choices(multiple_choices)
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)
            
            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=None,
                    inputs=[question_and_answers],
                    context=None,
                    references=ref,
                )
            )
        return input_examples