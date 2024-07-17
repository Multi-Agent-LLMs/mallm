import json
from pathlib import Path
from typing import Optional


def extract_draft(response: Optional[str]) -> Optional[str]:
    match_str = "final solution"
    if response:
        position = response.lower().rfind(match_str)
        if position == -1:
            return None
        matched_str = response[position + len(match_str) :].strip()
        if matched_str[0] == "]" or matched_str[0] == ":":
            matched_str = matched_str[1:].strip()
        return matched_str.split("\n\n")[
            0
        ]  # because LM tends to add extra explanation afterwards
    return None


def sort_output_file(input_file: str, output_file: str) -> None:
    """
    Sorts the output file to match the input file.
    """
    print(f"Sorting output file {output_file} to match the input file {input_file}...")
    with open(output_file) as file:
        data_out = json.load(file)
    with open(input_file) as file:
        data_in = json.load(file)

    # Create a dictionary to map example_ids to their corresponding data_out entries
    data_out_dict = {entry["exampleId"]: entry for entry in data_out}

    # Reorder data_out to match the order of example_ids in data_in
    sorted_data_out = [
        data_out_dict[entry["example_id"]]
        for entry in data_in
        if entry["example_id"] in data_out_dict
    ]

    Path(output_file).write_text(json.dumps(sorted_data_out, indent=4))
