from mallm.utils.config import Config
import re

config = Config(data=None, out=None, instruction=None)
attributes = {
    attr: getattr(config, attr)
    for attr in dir(config)
    if not callable(getattr(config, attr)) and not attr.startswith("__")
}

with open("README.md", "r+") as readme_file:
    content = readme_file.read()

    attributes_content = ""
    for attr, value in attributes.items():
        if isinstance(value, str):
            value = '"' + value + '"'
        attributes_content += f"{attr}: {type(value).__name__} = {value}\n"

    replacement_str = "### Config Arguments:\n```py\n" + attributes_content + "```"
    print(replacement_str)

    updated_lines = re.sub(
        r"### Config Arguments:\n```py\n(.*?)```",
        replacement_str,
        content,
        flags=re.DOTALL,
    )
    readme_file.seek(0)
    readme_file.writelines(updated_lines)
    readme_file.truncate()
