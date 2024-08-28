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


def run_configuration(
    config: Config, name: Optional[str], run_name: str, repeat: int
) -> None:
    original_out = ".".join(config.output_json_file_path.split(".")[:-1])
    config.output_json_file_path = f"{original_out}_repeat{repeat}.json"
    if name:
        config.output_json_file_path = f"{original_out}_{name}_repeat{repeat}.json"

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
        if config:
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
        print(f"  Data: {config.input_json_file_path}")
        print(f"  Out: {config.output_json_file_path}")
        print(f"  Model: {config.model_name}")
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
    name = config_data.get("name", None)

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
            run_configuration(deepcopy(config), name, f"Run {i}", repeat)

    print("\nBatch processing completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_mallm.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    run_batch(config_file)
