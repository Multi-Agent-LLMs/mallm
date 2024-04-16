import gzip
import json
import os
import shutil
import urllib
import uuid
import zipfile
from rdflib import Graph

from experiments.data.data_download import DatasetDownloader


def process_ttl_gz(path: str):
    print(f"Processing {path}")
    if os.path.exists(path + ".json"):
        file = open(path + ".json")
        data = json.load(file)
        return data
    g = Graph()
    g.parse(path + ".ttl")
    g.serialize(path + ".json", format="json-ld")
    f = open(path + ".json")
    data = json.load(f)
    return data


class EUCouncilVotesDownloader(DatasetDownloader):
    def custom_download(self):
        urllib.request.urlretrieve(
            "https://data.consilium.europa.eu/data/public-voting/council-votes-on-legislative-acts.zip",
            self.dataset_path + ".zip")
        with zipfile.ZipFile(self.dataset_path + ".zip", 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)
        with gzip.open(self.dataset_path + "/VotingResults.ttl.gz", 'rb') as f_in:
            with open(self.dataset_path + "/VotingResults.ttl", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Needed for future additions
        voting_results = process_ttl_gz(self.dataset_path + "/VotingResults")
        voting_results_cube = process_ttl_gz(self.dataset_path + "/VotingResultsCube")
        voting_results_new = process_ttl_gz(self.dataset_path + "/VotingResultsNew")

        votes = []
        definitions = [x for x in voting_results_cube if "http://www.w3.org/2004/02/skos/core#definition" in x]
        for act in definitions:
            act_id = act["@id"]
            votes = [x for x in voting_results_cube if
                     "http://data.consilium.europa.eu/data/public_voting/qb/dimensionproperty/act" in x and
                     "@id" in x["http://data.consilium.europa.eu/data/public_voting/qb/dimensionproperty/act"][0] and
                     x["http://data.consilium.europa.eu/data/public_voting/qb/dimensionproperty/act"][0][
                         "@id"] == act_id]
            if len(votes) == 0:
                continue
            votes_infavour = [vote for vote in votes if
                              vote["http://data.consilium.europa.eu/data/public_voting/qb/measureproperty/vote"][0][
                                  "@id"].split("/")[-1] == "votedinfavour"]
            votes_against = [vote for vote in votes if
                             vote["http://data.consilium.europa.eu/data/public_voting/qb/measureproperty/vote"][0][
                                 "@id"].split("/")[-1] == "votedagainst"]
            votes_abstained = [vote for vote in votes if
                               vote["http://data.consilium.europa.eu/data/public_voting/qb/measureproperty/vote"][0][
                                   "@id"].split("/")[-1] == "abstained"]

            print(
                f"Found {len(votes)} votes for {act_id} with {len(votes_infavour)} in favour, {len(votes_against)} against and {len(votes_abstained)} abstained ")
            print(act["http://www.w3.org/2004/02/skos/core#definition"][0]["@value"])
            print(f"{votes[0]["http://data.consilium.europa.eu/data/public_voting/qb/dimensionproperty/votingrule"][0]["@id"].split("/")[-1]}")

        # print(len(v))
        # print(v[0])
        # v = [x for x in v if "http://www.w3.org/2004/02/skos/core#definition" in x]
        # print(len(v))
        # print(v[0]["@type"])
        # v = [x for x in v if "http://data.consilium.europa.eu/data/public_voting/rdf/schema/Act" in x["@type"]]
        # print(len(v))

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
