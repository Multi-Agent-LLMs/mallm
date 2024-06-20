from mallm.utils.config import Config
import mallm.evaluation.evaluator as evaluator
from mallm.evaluation.evaluator import Evaluator
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
