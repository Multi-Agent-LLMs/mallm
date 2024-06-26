import json
import sys
from typing import Any, Optional

from mallm.scheduler import Scheduler
from mallm.utils.config import Config


def load_config(config_path: str) -> Any:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {config_path} is not a valid JSON file.")
        return {}
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return {}


def create_config(config_dict: Any) -> Optional[Config]:
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


def run_configuration(config: Config, run_name: str) -> None:
    if not validate_config(config):
        print(f"Skipping {run_name} due to validation error.")
        return

    try:
        print(f"Running {run_name}")
        scheduler = Scheduler(config)
        scheduler.run()
        print(f"Completed {run_name}")
    except Exception as e:
        print(f"Error running {run_name}: {e}")


def run_batch(config_path: str) -> None:
    config_data = load_config(config_path)
    if not config_data:
        print("No valid configuration found. Exiting.")
        return

    common_config = config_data.get("common", {})
    runs = config_data.get("runs", [])

    if not common_config:
        print("No common configuration found. Exiting.")
        return

    # Run specific configurations
    for i, run_config in enumerate(runs, 1):
        print(f"\nProcessing run {i}/{len(runs)}")
        # Merge common config with run-specific config, prioritizing run-specific values
        merged_config = {**common_config, **run_config}
        config = create_config(merged_config)
        if config:
            run_configuration(config, f"Run {i}")

    print("\nBatch processing completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_mallm.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    run_batch(config_file)
