import json
import sys
import traceback
from copy import deepcopy
from typing import Any, Optional, List, Dict

from mallm.scheduler import Scheduler
from mallm.utils.config import Config


def load_config(config_path: str) -> Any:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: {config_path} is not a valid JSON file.\n{e}")
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


def run_configuration(config: Config, run_name: str, name: str, repeat: int) -> None:
    # Adjust the output name for each repeat
    if config.out.startswith("."):
        config.out = config.out[1:]
    original_out = config.out.split(".")
    config.out = f"{original_out[0]}_{name}_repeat{repeat}.{original_out[1]}"

    try:
        print(f"Running {run_name} (Repeat {repeat})")
        scheduler = Scheduler(config)
        scheduler.run()
        print(f"Completed {run_name} (Repeat {repeat})")
    except Exception as e:
        print(f"Error running {run_name} (Repeat {repeat}): {e}")
        print(traceback.format_exc())


def validate_all_configs(
    common_config: Dict[str, Any], runs: List[Dict[str, Any]]
) -> List[Config]:
    valid_configs = []
    for i, run_config in enumerate(runs, 1):
        merged_config = {**common_config, **run_config}
        config = create_config(merged_config)
        if config and validate_config(config):
            valid_configs.append(config)
        else:
            print(f"Configuration for Run {i} is invalid.")
    return valid_configs


def summarize_runs(valid_configs: List[Config], repeats: int) -> None:
    print("\nRun Summary:")
    print(f"Total valid runs: {len(valid_configs)}")
    print(f"Repeats per run: {repeats}")
    print(f"Total executions: {len(valid_configs) * repeats}")
    print("\nValid Runs:")
    for i, config in enumerate(valid_configs, 1):
        print(f"Run {i}:")
        print(f"  Data: {config.data}")
        print(f"  Out: {config.out}")
        print(f"  Model: {config.model}")
        print(f"  Max Turns: {config.max_turns}")
        print()


def run_batch(config_path: str) -> None:
    config_data = load_config(config_path)
    if not config_data:
        print("No valid configuration found. Exiting.")
        return

    common_config = config_data.get("common", {})
    runs = config_data.get("runs", [])
    repeats = config_data.get("repeats", 1)
    name = config_data.get("name", "mallm")

    if not common_config:
        print("No common configuration found. Exiting.")
        return

    # Validate all configurations upfront
    valid_configs = validate_all_configs(common_config, runs)

    if len(valid_configs) != len(runs):
        print("Some configurations are invalid. Canceling batch process.")
        return

    # Summarize the runs
    summarize_runs(valid_configs, repeats)

    print("Starting batch processing.")

    # Run valid configurations
    for i, config in enumerate(valid_configs, 1):
        print(f"\nProcessing run {i}/{len(valid_configs)}")
        for repeat in range(1, repeats + 1):
            run_configuration(deepcopy(config), f"Run {i}", name, repeat)

    print("\nBatch processing completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_mallm.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    run_batch(config_file)
