<br />
<p align="center">
<a><img src="image/mallm.webp" alt="MALLM" width="128" height="128" title="MALLM"></a>
  <h3 align="center">MALLM</h3>
  <p align="center">
    Multi-Agent LLMs For Conversational Task-Solving: Framework<br />
    <p align="center">
  <a href="https://github.com/Multi-Agent-LLMs/mallm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Multi-Agent-LLMs/mallm" alt="License"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a>
  <a href="https://codecov.io/gh/Multi-Agent-LLMs/mallm"><img src="https://codecov.io/gh/Multi-Agent-LLMs/mallm/graph/badge.svg?token=CBUZTPV5KA" alt="Coverage"></a>
  <a href="https://github.com/Multi-Agent-LLMs/mallm/actions/workflows/python-package.yml"><img src="https://github.com/Multi-Agent-LLMs/mallm/actions/workflows/python-package.yml/badge.svg" alt="Pipeline"></a>
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

Create an environment with:
`conda create --name mallm python=3.12`

### Package
Install as a package:
`pip install -e .`

### Test Data
Create the test data
`python data/data_downloader.py`

You also need the checkpoints for the LLM you want to use. Use instruction-tuned models for the best performance.

### Run from Terminal
MALLM relies on an external API like OpenAI or Text Generation Inference by Huggingface.
Check the information [here (tg-hpc)](https://github.com/Multi-Agent-LLMs/tgi-hpc) or [here (tgi-scc)](https://github.com/Multi-Agent-LLMs/tgi-scc) about how to host a model yourself.

Once the endpoint is available, you can initiate all discussions by a single script. Example with TGI:

`python mallm/scheduler.py --data=data/datasets/etpc_debugging.json --out=test_out.json --instruction="Paraphrase the input text." --endpoint_url="http://127.0.0.1:8080" --model="tgi"`

Or with OpenAI:

`python mallm/scheduler.py --data=data/datasets/etpc_debugging.json --out=test_out.json --instruction="Paraphrase the input text." --endpoint_url="https://api.openai.com" --model="gpt-3.5-turbo" --api_key="<your-key>"`

## Run as Module
If installed, you can use MALLM from anywhere on your system:
```py
from mallm import scheduler
from mallm.utils.config import Config

mallm_scheduler = scheduler.Scheduler(
  Config(
    data="data/datasets/etpc_debugging.json",
    out="test_out.json",
    instruction="Paraphrase the input text.",
    endpoint_url="http://127.0.0.1:8080",
    model="tgi"
  )
)
mallm_scheduler.run()
```

You can also call the API from OpenAI:
```py
mallm_scheduler = scheduler.Scheduler(
  Config(
    data="data/datasets/etpc_debugging.json",
    out="test_out.json",
    instruction="Paraphrase the input text.",
    endpoint_url="https://api.openai.com",
    model="gpt-3.5-turbo", # or another model from this list: https://platform.openai.com/docs/models
    api_key="<your-key>"
  )
)
```

## Project Structure

MALLM is composed of three parts:
The framework follows this structure and can be found in the `mallm` directory.

1) Agents (subdirectory: `mallm/agents/`)
2) Discourse Policy (subdirectory: `mallm/discourse_policy/`)
3) Decision Protocol (subdirectory: `mallm/decision_protocol/`)

Experiments can be implemented as a seperate repository, loading MALLM as a package.
You can test stuff in the `notebooks` directory.

## Arguments

Use "tgi" as a model for Text Generation Inference by HuggingFace or one of these: https://platform.openai.com/docs/models

### Config Arguments:
```py
agent_generator: str = "expert"
api_key: str = "-"
baseline: bool = False
chain_of_thought: bool = True
clear_memory_bucket: bool = True
context_length: int = 3
data: NoneType = None
debate_rounds: NoneType = None
decision_protocol: str = "hybrid_consensus"
endpoint_url: str = "https://api.openai.com"
extract_all_drafts: bool = True
feedback_sentences: NoneType = None
force_all_turns: bool = False
include_current_turn_in_memory: bool = True
instruction: NoneType = None
max_concurrent_requests: int = 100
max_turns: int = 10
memory_bucket_dir: str = "./mallm/utils/memory_bucket/"
model: str = "gpt-3.5-turbo"
num_agents: int = 3
num_samples: NoneType = None
out: NoneType = None
paradigm: str = "memory"
split_agree_and_answer: bool = True
use_moderator: bool = False
```

## Evaluation

We provide some basic evaluation metrics that can be directly applied to the output json of mallm.
Supported metrics: `bertscore`, `bleu`, `meteor`, `multichoice`, `rouge`

From terminal:

`python mallm/evaluation/evaluator.py --input_file_path="test_out.json" --output_file_path="test_out_evaluated.json" --metrics=[bleu,rouge]`

From script:

```py
from mallm.evaluation.evaluator import Evaluator

evaluator = Evaluator(input_file_path= "test_out.json", output_file_path ="test_out_evaluated.json", metrics = ["bleu","rouge"])
evaluator.process()
```

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

## Using the Batch Executor

The batch executor allows you to run multiple configurations of the MALLM (Multi-Agent Language Model) scheduler in sequence. This is useful for running experiments with different parameters or processing multiple datasets.

### Location
- The batch executor script is located in the `notebook` folder and is named `batch_mallm.py`.
- A template for the batch configuration file is provided as `batch.json.template` in the same folder.

### Setup

1. **Prepare your configuration file:**
   - Copy the `batch.json.template` file and rename it (e.g., `my_batch_config.json`).
   - Edit the JSON file to define your configurations. Each object in the array represents a separate run with its own parameters.

   Example:
   ```json
   [
     {
       "data": "path/to/data1.csv",
       "out": "path/to/output1.csv",
       "instruction": "Instruction for run 1",
       "model": "gpt-3.5-turbo",
       "max_turns": 10
     },
     {
       "data": "path/to/data2.csv",
       "out": "path/to/output2.csv",
       "instruction": "Instruction for run 2",
       "model": "gpt-4",
       "max_turns": 15
     }
   ]
   ```

2. **Ensure all required dependencies are installed.**

### Running the Batch Executor

To run the batch executor, use the following command from the terminal:

```
python notebook/batch_mallm.py path/to/your/config_file.json
```

Replace `path/to/your/config_file.json` with the actual path to your JSON configuration file.

### Behavior

- The batch executor will process each configuration in the order they appear in the JSON file.
- For each configuration:
  - It will create a `Config` object with the specified parameters.
  - It will then initialize a `Scheduler` with this configuration and run it.
  - Progress and any errors will be printed to the console.
- If a configuration is invalid or encounters an error during execution, the batch processor will skip to the next configuration.
- The process continues until all configurations have been attempted.

### Error Handling

- The batch executor is designed to be robust against various types of errors:
  - Invalid JSON configuration files
  - Missing required fields in a configuration
  - Errors during the execution of individual configurations
- Any errors encountered will be reported in the console output, but will not stop the entire batch process.

### Tips

- Always test your configurations individually before running them in a batch to ensure they work as expected.
- Use descriptive output file names to easily identify the results of each run.
- Monitor the console output for any error messages or skipped configurations.

By using the batch executor, you can easily run multiple experiments or process various datasets without manual intervention, saving time and reducing the chance of configuration errors.

## Contributing
If you want to contribute, please use this pre-commit hook to ensure the same formatting for everyone.
```bash
pip install pre-commit
pre-commit install
```

### Testing
You can run unit tests locally:
`pytest ./test/`
