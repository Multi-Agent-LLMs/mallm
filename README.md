<br />
<p align="center">
<a><img src="image/mallm.webp" alt="MALLM" width="128" height="128" title="FawnRescue"></a>
  <h3 align="center">MALLM</h3>
  <p align="center">
    Multi-Agent LLMs For Conversational Task-Solving: Framework<br />
    <p align="center">
  <a href="https://github.com/Multi-Agent-LLMs/mallm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Multi-Agent-LLMs/mallm" alt="License"></a>
  <a href="https://github.com/Multi-Agent-LLMs/mallm/network/members"><img src="https://img.shields.io/github/forks/Multi-Agent-LLMs/mallm?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/Multi-Agent-LLMs/mallm/stargazers"><img src="https://img.shields.io/github/stars/Multi-Agent-LLMs/mallm?style=social" alt="GitHub stars"></a>
</p>
    <p>
    <a href="https://github.com/Multi-Agent-LLMs/mallm/issues">Report Bug</a>
    ·
    <a href="https://github.com/Multi-Agent-LLMs/mallm/issues">Request Feature</a>
    </p>
  </p>
</p>

## Install

Create an environment with
`conda env create -f environment.yml`

### Package
Install as a package
`pip install -e .`

### Test Data
Create the test data
`python data/data_downloader.py`

You also need the checkpoints for the LLM you want to use. Currently, LLaMA-2-70b-chat has been tested and is working.

### Run Discussions
MALLM relies on an external API (Text Generation Inference by Huggingface).
Check the information [here (tg-hpc)](https://github.com/Multi-Agent-LLMs/tgi-hpc) or [here (tgi-scc)](https://github.com/Multi-Agent-LLMs/tgi-scc) about how to host a model yourself.

Once the endpoint is available, you can initiate all discussions by a single script. Example:

`python mallm/run_async.py --data=data/datasets/etpc_debugging.json --out=test_out.json --instruction="Paraphrase the input text." --endpoint_url="http://127.0.0.1:8080" --hf_api_token="YOUR_TOKEN" --max_concurrent_requests=100`

While each discussion is sequential, multiple discussions can be processed in parallel for significant speedup. Please set `max_concurrent_requests` to a reasonable number so that you do not block the GPU for all other users of the TGI instance.

More parameters:
```
use_moderator=False,
max_turns=10,
feedback_sentences=[3, 4],
paradigm="memory",
context_length=1,
include_current_turn_in_memory=False,
max_concurrent_requests=100,
```

## Project Structure

MALLM is composed of three parts:
The framework follows this structure and can be found in the `mallm` directory.

1) Agents (subdirectory: `mallm/agents/`)
2) Discourse Policy (subdirectory: `mallm/discourse_policy/`)
3) Decision Making (subdirectory: `mallm/decision_making/`)

Experiments can be implemented as a seperate repository, loading MALLM as a package.
You can test stuff in the `notebooks` directory.

Please do not develop on master and create a branch. Thank you!

## Logging

To enable logging you can add a handler to the library logger. This can be done with the following code

```py
import logging

# Configure logging for the library
library_logger = logging.getLogger("mallm")
library_logger.setLevel(logging.INFO)

# Add handlers to the logger
stream_handler = logging.StreamHandler()

# Optionally set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# Attach the handler to the logger
library_logger.addHandler(stream_handler)
```

## Building
If you want to build the package locally (not from PyPI), you can use these commands in the root directory on an up-to-date `pyproject.toml` file.
You can also use [this link](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for help.
```bash
py -m pip install --upgrade build
py -m build
```
Then install the wheel from the `dist` directory.
```bash
pip install ./dist/mallm-version.tar.gz
```

## Contributing
If you want to contribute, please use this pre-commit hook to ensure the same formatting for everyone.
```bash
pip install pre-commit
pre-commit install
```
