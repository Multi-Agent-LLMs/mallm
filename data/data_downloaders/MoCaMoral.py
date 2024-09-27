import requests
import json
import uuid
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample

class MoCaMoralDownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(
        self, sample_size: Optional[int] = None, hf_token: Optional[str] = None
    ):
        super().__init__(
            name="moca_moral",
            dataset_name="MoCa-Moral",
            version="default",
            sample_size=sample_size,
            hf_dataset=False,
            hf_token=hf_token,
        )

    def custom_download(self):
        github_link = "https://raw.githubusercontent.com/cicl-stanford/moca/refs/heads/main/data/moral_dataset_v1.json"
        response = requests.get(github_link)
        if response.status_code == 200:
            self.dataset = response.json()
        else:
            print(f"Failed to download data from {github_link}. Status code: {response.status_code}")

    def process_data(self) -> list[InputExample]:
        input_examples = []

        for i, sample in enumerate(self.dataset[: self.sample_size]):
            sample = json.loads(sample)
            formatted_answers = self._format_answer_choices(["Yes", "No"])
            question_and_answers = f"{sample["question"]}\n\n" + "\n".join(formatted_answers)

            metadata = {
                "answerDist": sample["answer_dist"],
                "annotatedSentences": sample["annotated_sentences"],
            }

            input_examples.append(
                InputExample(
                    example_id=str(uuid.uuid4()),
                    dataset_id=str(i),
                    inputs=[question_and_answers],
                    context=[sample["story"]],
                    references=[sample["answer"].replace("Yes", "A) Yes").replace("No", "B) No")],
                    metadata=metadata,
                )
            )
        return input_examples