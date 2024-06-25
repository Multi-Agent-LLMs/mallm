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
Download and create the test data: `python data/data_downloader.py --datasets=[SQuAD2,ETPC] --sample_size=100`

You can use any dataset for this project as long as it follows [this basic format](https://github.com/Multi-Agent-LLMs/mallm/blob/main/data/datasets/etpc_debugging.json). These datasets are supported by our automated formatting pipeline: `Multi-News`, `ETPC`, `WMT19_de_en`, `XSum`, `Europarl`, `BTVote`, `StrategyQA`, `SQuAD2`, `GSM8K`, `GPQA`, `SimpleEthicalQuestions`

### Run from Terminal
MALLM relies on an external API like OpenAI or Text Generation Inference by Huggingface.
Check the information [here (tg-hpc)](https://github.com/Multi-Agent-LLMs/tgi-hpc) or [here (tgi-scc)](https://github.com/Multi-Agent-LLMs/tgi-scc) about how to host a model yourself.
For self-hosting you need the checkpoints for the instruction-tuned model you want to use.

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
response_generator: str = "json"
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

## Contributing
If you want to contribute, please use this pre-commit hook to ensure the same formatting for everyone.
```bash
pip install pre-commit
pre-commit install
```

### Testing
You can run unit tests locally:
`pytest ./test/`
