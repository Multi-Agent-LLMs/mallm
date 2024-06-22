from mallm.utils.config import Config
from pathlib import Path
import mallm.evaluation.evaluator as evaluator
import re

# SCHEDULER CONFIG
print("Updating Scheduler Config...")
with open("README.md", "r+") as readme_file:
    config = Config(data=None, out=None, instruction=None)
    attributes = {
        attr: getattr(config, attr)
        for attr in dir(config)
        if not callable(getattr(config, attr)) and not attr.startswith("__")
    }

    content = readme_file.read()

    attributes_content = ""
    for attr, value in attributes.items():
        if isinstance(value, str):
            value = '"' + value + '"'
        attributes_content += f"{attr}: {type(value).__name__} = {value}\n"

    replacement_str = "### Config Arguments:\n```py\n" + attributes_content + "```"

    updated_lines = re.sub(
        r"### Config Arguments:\n```py\n(.*?)```",
        replacement_str,
        content,
        flags=re.DOTALL,
    )
    readme_file.seek(0)
    readme_file.writelines(updated_lines)
    readme_file.truncate()

# EVALUATOR
print("Updating Evaluator Metrics...")
with open("README.md", "r+") as readme_file:
    content = readme_file.read()

    metrics = [metric._name for metric in evaluator.ALL_METRICS]
    metrics_str = ""
    for m in metrics:
        metrics_str += "`" + m.lower() + "`, "
    metrics_str = metrics_str[:-2]
    replacement_str = "Supported metrics: " + metrics_str + "\n"

    updated_lines = re.sub(
        r"Supported metrics: (.*?)\n",
        replacement_str,
        content,
        flags=re.DOTALL,
    )
    readme_file.seek(0)
    readme_file.writelines(updated_lines)
    readme_file.truncate()

# DATASETS
print("Updating Supported Datasets...")
with open("README.md", "r+") as readme_file:
    content = readme_file.read()

    supported_datasets = ""
    for file in Path("data/data_downloaders/").glob("*.py"):
        if not file.name == "__init__.py":
            supported_datasets += f"`{file.name.split(".")[0]}`, "
    supported_datasets = supported_datasets[:-2]
    replacement_str = (
        "These datasets are supported by our automated formatting pipeline: "
        + supported_datasets
        + "\n"
    )

    updated_lines = re.sub(
        r"These datasets are supported by our automated formatting pipeline: (.*?)\n",
        replacement_str,
        content,
        flags=re.DOTALL,
    )
    readme_file.seek(0)
    readme_file.writelines(updated_lines)
    readme_file.truncate()
