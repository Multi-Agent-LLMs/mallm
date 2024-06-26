import json
import sys
from typing import List, Dict, Any, Optional

from mallm.scheduler import Scheduler
from mallm.utils.config import Config


def load_configs(config_path: str) -> List[Dict[str, Any]]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {config_path} is not a valid JSON file.")
        return []
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return []


def create_config(config_dict: Dict[str, Any]) -> Optional[Config]:
    try:
        return Config(**config_dict)
    except TypeError as e:
        print(f"Error creating Config object: {e}")
        return None


def validate_config(config: Config) -> bool:
    required_fields = ["data", "out", "instruction"]
    for field in required_fields:
        if not getattr(config, field):
            print(f"Error: '{field}' is required but not provided or empty.")
            return False
    return True


def run_batch(config_file: str):
    configs = load_configs(config_file)
    if not configs:
        print("No valid configurations found. Exiting.")
        return

    for i, config_dict in enumerate(configs):
        print(f"\nProcessing configuration {i + 1}/{len(configs)}")
        config = create_config(config_dict)
        if config is None:
            print(f"Skipping configuration {i + 1} due to error.")
            continue

        if not validate_config(config):
            print(f"Skipping configuration {i + 1} due to validation error.")
            continue

        try:
            print(f"Running configuration {i + 1}")
            scheduler = Scheduler(config)
            scheduler.run()
            print(f"Completed configuration {i + 1}")
        except Exception as e:
            print(f"Error running configuration {i + 1}: {e}")
            print(f"Skipping to next configuration.")

    print("\nBatch processing completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_executor.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    run_batch(config_file)
