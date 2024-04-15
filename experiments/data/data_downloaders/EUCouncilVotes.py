import gzip
import json
import shutil
import urllib
import uuid
import zipfile
from rdflib import Graph

from experiments.data.data_download import DatasetDownloader


class EUCouncilVotesDownloader(DatasetDownloader):
    def custom_download(self):
        urllib.request.urlretrieve(
            "https://data.consilium.europa.eu/data/public-voting/council-votes-on-legislative-acts.zip",
            self.dataset_path + ".zip")
        with zipfile.ZipFile(self.dataset_path + ".zip", 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)
        with gzip.open(self.dataset_path + "/VotingResultsNew.ttl.gz", 'rb') as f_in:
            with open(self.dataset_path + "/VotingResultsNew.ttl", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        g = Graph()
        g.parse(self.dataset_path + "/VotingResultsNew.ttl")
        v = g.serialize(self.dataset_path + "/VotingResultsNew.json", format="json-ld")
        v = g.serialize(self.dataset_path + "/VotingResultsNew.xml", format="xml")
        print(v)

    def __init__(self):
        super().__init__('eu_council_votes', hf_dataset=False)

    def process_data(self):
        json_str = ""
        for s in self.dataset[:self.sample_size]:
            ref = [k for k, v in s["target_scores"].items() if v == 1]
            multiple_choice_str = " Answer Choices:"
            for i, (k, v) in enumerate(s["target_scores"].items()):
                multiple_choice_str += " " + f"{chr(ord('A') + i)}) " + k
            json_str += f'''{{ "exampleId":"{str(uuid.uuid4())}", "datasetId": null, "input":{json.dumps(s['input'] + multiple_choice_str)}, "context": null, "references": [{json.dumps(ref[0])}], "personas": null }}\n'''
        self.save_to_json(json_str)
