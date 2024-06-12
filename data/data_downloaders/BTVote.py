import json
import os
import uuid

import pandas as pd
import requests
import random
from typing import Optional

from data.data_download import DatasetDownloader
from mallm.utils.types import InputExample


def list_files(dataset_persistent_id):
    base_url = "https://dataverse.harvard.edu/api/datasets/:persistentId"
    params = {
        "persistentId": dataset_persistent_id,
    }
    response = requests.get(f"{base_url}/versions/:latest/files", params=params)
    file_ids = []
    if response.status_code == 200:
        data = response.json()
        for item in data["data"]:
            file_ids.append(item["dataFile"]["id"])
    else:
        print("Failed to retrieve files. Status code:", response.status_code)
    return file_ids


def download_file(file_id, filename):
    # The correct base URL and endpoint to download a file using its file ID
    download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"

    # Making the GET request to download the file
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


class BTVoteDownloader(DatasetDownloader):
    def custom_download(self):
        file_ids_vote_behaviour = list_files("doi:10.7910/DVN/24U1FR")
        file_ids_characteristics = list_files("doi:10.7910/DVN/QSFXLQ")
        file_ids_vote_characteristics = list_files("doi:10.7910/DVN/AHBBXY")

        if not os.path.exists("data/datasets/btvote"):
            os.mkdir("data/datasets/btvote")

        download_file(file_ids_vote_behaviour[1], "data/datasets/btvote/behaviour.dta")
        download_file(
            file_ids_characteristics[1], "data/datasets/btvote/characteristics.tab"
        )
        download_file(
            file_ids_vote_characteristics[0],
            "data/datasets/btvote/vote_characteristics.tab",
        )

        data_behaviour = pd.read_stata("data/datasets/btvote/behaviour.dta")
        data_characteristics = pd.read_csv(
            "data/datasets/btvote/characteristics.tab", delimiter="\t", encoding="utf-8"
        )
        data_vote_characteristics = pd.read_csv(
            "data/datasets/btvote/vote_characteristics.tab",
            delimiter="\t",
            encoding="utf-8",
        )

        self.dataset = {
            "behaviour": data_behaviour,
            "characteristics": data_characteristics,
            "vote_characteristics": data_vote_characteristics,
        }

    def __init__(self, sample_size: Optional[int], hf_token: Optional[str] = None):
        super().__init__("btvote", hf_dataset=False, sample_size=sample_size)

    def process_data(self) -> list[InputExample]:
        merged_df = pd.merge(
            self.dataset["behaviour"],
            self.dataset["vote_characteristics"],
            on="vote_id",
            how="inner",
        )
        grouped = (
            merged_df.groupby(["vote_title", "vote_beh"], observed=False)
            .size()
            .reset_index(name="counts")
        )
        pivot_table = grouped.pivot(
            index="vote_title", columns="vote_beh", values="counts"
        ).fillna(0)

        pivot_table = pivot_table.reset_index()
        input_examples = []
        for idx, row in pivot_table.iterrows():
            example_id = str(uuid.uuid4())
            input_data = row["vote_title"]
            reference_data = json.dumps(
                {col: row[col] for col in pivot_table.columns if col != "vote_title"}
            )
            input_examples.append(
                InputExample(
                    example_id=example_id,
                    dataset_id=None,
                    inputs=[input_data],
                    context=None,
                    references=[reference_data],
                    personas=None,
                )
            )
        random.shuffle(input_examples)
        input_examples = input_examples[: self.sample_size]
        return input_examples
