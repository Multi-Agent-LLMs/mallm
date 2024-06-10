import dataclasses
import importlib
import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import fire
from datasets import load_dataset
from mallm.utils.types import InputExample


class DatasetDownloader(ABC):
    def __init__(
        self,
        name: str,
        hf_dataset: bool = True,
        dataset_name: str = None,
        version: str = "main",
        sample_size: int = None,
        trust_remote_code: bool = False,
    ):
        self.name = name
        self.version = version
        self.dataset_path = f"data/datasets/{name}"
        self.output_path = f"data/datasets/{name}.json"
        self.dataset = None
        self.sample_size = sample_size
        if dataset_name is None:
            self.dataset_name = name
        else:
            self.dataset_name = dataset_name
        self.hf_dataset = hf_dataset
        self.trust_remote_code = trust_remote_code

    def download(self) -> bool:
        if not os.path.exists(self.output_path):
            print(f"\n\033[94m[INFO]\033[0m Downloading {self.name}...")
            if self.hf_dataset:
                self.hf_download()
            else:
                self.custom_download()
            return True
        else:
            print(f"\033[92m[INFO]\033[0m {self.name} already downloaded.")
            return False

    def hf_download(self):
        self.dataset = load_dataset(
            self.dataset_name, self.version, trust_remote_code=self.trust_remote_code
        )
        self.dataset.save_to_disk(self.dataset_path)

    @abstractmethod
    def custom_download(self):
        pass

    @abstractmethod
    def process_data(self) -> list[InputExample]:
        pass

    def save_to_json(self, data: list[InputExample]):
        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(json.dumps([dataclasses.asdict(example) for example in data]))
        print(f"\033[93m[INFO]\033[0m Data saved to {self.output_path}")

    def shuffle_and_select(self, split="train"):
        if self.dataset:
            if self.sample_size:
                return (
                    self.dataset[split].shuffle(seed=42).select(range(self.sample_size))
                )
            else:
                return self.dataset[split].shuffle(seed=42)
        else:
            raise ValueError("Dataset not loaded.")


def find_downloader_classes(module):
    for attribute_name in dir(module):
        if attribute_name == "DatasetDownloader":
            continue
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(
            attribute, module.DatasetDownloader
        ):
            return attribute
    return None


def load_and_execute_downloaders(directory="data_downloaders", datasets=None):
    # Path to the directory containing downloader modules
    base_path = Path(__file__).parent / directory
    # Iterate over each file in the directory
    for file in base_path.glob("*.py"):
        if (
            file.name == "__init__.py"
            or datasets
            and file.name.split(".")[0] not in datasets
        ):
            continue
        print(f"\n\033[96m[PROCESSING]\033[0m Processing {file.name}")
        module_name = f"{directory}.{file.stem}"
        # Dynamically import the module
        module = importlib.import_module(module_name)
        # Find the downloader class
        downloader_class = find_downloader_classes(module)
        if downloader_class:
            # Instantiate and execute the downloader
            downloader = downloader_class()
            try:
                if downloader.download():
                    input_examples = downloader.process_data()
                    for example in input_examples:
                        example.confirm_types()
                    downloader.save_to_json(input_examples)
                print(f"\033[92m[COMPLETED]\033[0m Processed {downloader.name}")
            except Exception as e:
                print(f"\033[91m[ERROR]\033[0m Error processing {downloader.name}: {e}")
                traceback.print_exc()
        else:
            print(f"\033[93m[WARNING]\033[0m No downloader class found in {file.name}")
        print("\033[90m" + "-" * 40 + "\033[0m")  # Separator for clarity


if __name__ == "__main__":
    fire.Fire(load_and_execute_downloaders)
