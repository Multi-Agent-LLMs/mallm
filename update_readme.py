from mallm.utils.config import Config
from pathlib import Path
import mallm.evaluation.evaluator as evaluator
import typing
from mallm.utils.dicts import (
    DECISION_PROTOCOLS,
    DISCUSSION_PARADIGMS,
    PERSONA_GENERATORS,
    RESPONSE_GENERATORS,
)
import re

# SCHEDULER CONFIG
print("Updating Scheduler Config...")
with open("README.md", "r+") as readme_file:
    config = Config(input_json_file_path=None, output_json_file_path=None, task_instruction_prompt=None)
    attributes = {
        attr: getattr(config, attr)
        for attr in dir(config)
        if not callable(getattr(config, attr)) and not attr.startswith("__")
    }

    content = readme_file.read()

    types = typing.get_type_hints(Config, include_extras=True)
    attributes_content = ""
    for attr, type in types.items():
        value = attributes[attr]
        if isinstance(value, str):
            value = '"' + value + '"'
        type_str = str(type).replace("typing.", "")
        if "Optional" in type_str:
            type_str = type_str.replace("Optional[", "", 1).rsplit("]", 1)[0]
            attributes_content += f"{attr}: Optional[{type_str}] = {value}\n"
        else:
            attributes_content += f"{attr}: {type.__name__} = {value}\n"

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

    metrics = sorted([metric._name.lower() for metric in evaluator.ALL_METRICS])
    metrics_str = ""
    for m in metrics:
        metrics_str += "`" + m + "`, "
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
    for file in sorted(Path("data/data_downloaders/").glob("*.py")):
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

# DISCUSSION PARAMETERS:
print("Updating Discussion Parameters...")
with open("README.md", "r+") as readme_file:
    config = Config(input_json_file_path=None, output_json_file_path=None, task_instruction_prompt=None)
    content = readme_file.read()

    extra_content = "\nResponse Generators: "
    for key in sorted(RESPONSE_GENERATORS.keys()):
        extra_content += "`" + key + "`, "
    extra_content = extra_content[:-2]
    extra_content += "\n\nDecision Protocols: "
    for key in sorted(DECISION_PROTOCOLS.keys()):
        extra_content += "`" + key + "`, "
    extra_content = extra_content[:-2]
    extra_content += "\n\nPersona Generators: "
    for key in sorted(PERSONA_GENERATORS.keys()):
        extra_content += "`" + key + "`, "
    extra_content = extra_content[:-2]
    extra_content += "\n\nDiscussion Paradigms: "
    for key in sorted(DISCUSSION_PARADIGMS.keys()):
        extra_content += "`" + key + "`, "
    extra_content = extra_content[:-2]

    replacement_str = "### Discussion Parameters:" + extra_content + "\n\n## Evaluation"

    updated_lines = re.sub(
        r"### Discussion Parameters:(.*?)\n\n## Evaluation",
        replacement_str,
        content,
        flags=re.DOTALL,
    )
    readme_file.seek(0)
    readme_file.writelines(updated_lines)
    readme_file.truncate()
