import json
import urllib
import uuid
import random

import pandas as pd
import requests

from data.data_download import DatasetDownloader


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
            print(
                f"File ID: {item['dataFile']['id']}, File Name: {item['dataFile']['filename']}"
            )
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
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


class BTVote(DatasetDownloader):
    def custom_download(self):
        file_ids_vote_behaviour = list_files("doi:10.7910/DVN/24U1FR")
        file_ids_characteristics = list_files("doi:10.7910/DVN/QSFXLQ")
        file_ids_vote_characteristics = list_files("doi:10.7910/DVN/AHBBXY")

        download_file(file_ids_vote_behaviour[1], "data/behaviour.dta")
        download_file(file_ids_characteristics[1], "data/characteristics.tab")
        download_file(file_ids_vote_characteristics[0], "data/vote_characteristics.tab")

        data_behaviour = pd.read_stata("data/behaviour.dta")
        data_characteristics = pd.read_csv(
            "data/characteristics.tab", delimiter="\t", encoding="ISO-8859-1"
        )
        data_vote_characteristics = pd.read_csv(
            "data/vote_characteristics.tab", delimiter="\t", encoding="ISO-8859-1"
        )

        self.dataset = {
            "behaviour": data_behaviour,
            "characteristics": data_characteristics,
            "vote_characteristics": data_vote_characteristics,
        }

    def __init__(self):
        super().__init__("btvote", hf_dataset=False)

    def process_data(self):
        pass
