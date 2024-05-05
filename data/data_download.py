import os
import importlib
import sys
from abc import abstractmethod, ABC
from pathlib import Path
from datasets import load_dataset


class DatasetDownloader(ABC):
    def __init__(
        self,
        name: str,
        hf_dataset: bool = True,
        dataset_name: str = None,
        version: str = "main",
        sample_size: int = 1000,
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
            print(f"Downloading {self.name}...")
            if self.hf_dataset:
                self.hf_download()
            else:
                self.custom_download()
            return True
        else:
            print(f"{self.name} already downloaded.")
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
    def process_data(self):
        pass

    def save_to_json(self, data):
        with open(self.output_path, "w") as file:
            file.write(data)

    def shuffle_and_select(self, split="train"):
        if self.dataset:
            return self.dataset[split].shuffle(seed=42).select(range(self.sample_size))
        else:
            raise ValueError("Dataset not loaded.")


def find_downloader_classes(module):
    for attribute_name in dir(module):
        if attribute_name == "DatasetDownloader":
            continue
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and "Downloader" in attribute_name:
            return attribute
    return None


def load_and_execute_downloaders(directory="data_downloaders"):
    # Path to the directory containing downloader modules
    base_path = Path(__file__).parent / directory
    # Iterate over each file in the directory
    for file in base_path.glob("*.py"):
        if file.name == "__init__.py":
            continue
        print(f"Processing {file.name}")
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
                    downloader.process_data()
                print(f"Processed {downloader.name}")
            except Exception as e:
                sys.stderr.write(f"Error processing {downloader.name}:\n{e}\n")


if __name__ == "__main__":
    load_and_execute_downloaders()
