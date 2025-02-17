<br />
<p align="center">
<a><img src="https://raw.githubusercontent.com/Multi-Agent-LLMs/mallm/refs/heads/main/image/mallm.webp" alt="MALLM" width="128" height="128" title="MALLM"></a>
  <h3 align="center">MALLM</h3>
  <p align="center">
    Multi-Agent LLMs For Conversational Task-Solving: Framework<br />
    <p align="center">
  <a href="https://arxiv.org/abs/2410.22932"><img src="https://img.shields.io/badge/arXiv-2410.22932-red" alt="arXiv"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a>
  <a href="https://github.com/Multi-Agent-LLMs/mallm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Multi-Agent-LLMs/mallm" alt="License"></a>
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

### Create Data
Download and create the test data: `python data/data_downloader.py --datasets=[SQuAD2,ETPC]`

You can use any dataset for this project as long as it follows [this basic format](https://github.com/Multi-Agent-LLMs/mallm/blob/main/data/datasets/etpc_debugging.json). These datasets are supported by our automated formatting pipeline: `AquaRat`, `BBQGenderIdentity`, `BTVote`, `ETHICS`, `ETPC`, `Europarl`, `GPQA`, `GSM8K`, `IFEval`, `MMLU`, `MMLUPro`, `MUSR`, `MathLvl5`, `MoCaMoral`, `MoralExceptQA`, `MultiNews`, `SQuAD2`, `SimpleEthicalQuestions`, `StrategyQA`, `WMT19DeEn`, `WinoGrande`, `XSum`

### Run from Terminal
MALLM relies on an external API like OpenAI or Text Generation Inference by Huggingface.

Once the endpoint is available, you can initiate all discussions with a single script. Example with TGI:

`python mallm/scheduler.py --input_json_file_path=data/datasets/etpc_debugging.json --output_json_file_path=test_out.json --task_instruction_prompt="Paraphrase the input text." --endpoint_url="http://127.0.0.1:8080/v1"`

Or with OpenAI:

`python mallm/scheduler.py --input_json_file_path=data/datasets/etpc_debugging.json --output_json_file_path=test_out.json --task_instruction_prompt="Paraphrase the input text." --endpoint_url="https://api.openai.com/v1" --api_key="<your-key>"`

## Run command line scripts
You can run the command line scripts from the terminal. The following command will run the scheduler with the given parameters:

`mallm-run --input_json_file_path=data/datasets/etpc_debugging.json --output_json_file_path=test_out.json --task_instruction_prompt="Paraphrase the input text." --endpoint_url="http://127.0.0.1:8080/v1" --model_name="tgi"`

or use the evaluation script:

`mallm-evaluate --input_json_file_path=test_out.json --output_json_file_path=test_out_evaluated.json --metrics=[bleu,rouge]`

## Run as Module
If installed, you can use MALLM in code:

```py
from mallm import scheduler
from mallm.utils.config import Config

mallm_scheduler = scheduler.Scheduler(
    Config(
        input_json_file_path="data/datasets/etpc_debugging.json",
        output_json_file_path="test_out.json",
        task_instruction_prompt="Paraphrase the input text.",
        endpoint_url="http://127.0.0.1:8080/v1"
    )
)
mallm_scheduler.run()
```

## Code Structure

MALLM is composed of three parts.
The framework follows this structure and can be found in the `mallm` directory.

1) Agents (subdirectory: `mallm/agents/`)
2) Discourse Policy (subdirectory: `mallm/discourse_policy/`)
3) Decision Protocol (subdirectory: `mallm/decision_protocol/`)

Experiments can be implemented as a separate repository, loading MALLM as a package.

## Arguments

### Config Arguments:
```py
input_json_file_path: str = None
output_json_file_path: str = None
task_instruction_prompt: str = None
task_instruction_prompt_template: Optional[str] = None
endpoint_url: str = "https://api.openai.com/v1"
model_name: str = "gpt-3.5-turbo"
api_key: str = "-"
max_turns: int = 10
skip_decision_making: bool = False
discussion_paradigm: str = "memory"
response_generator: str = "simple"
decision_protocol: str = "hybrid_consensus"
visible_turns_in_memory: int = 2
debate_rounds: int = 2
concurrent_api_requests: int = 100
use_baseline: bool = False
use_chain_of_thought: bool = True
num_agents: int = 3
num_neutral_agents: int = 0
agent_generator: str = "expert"
agent_generators_list: list = []
trust_remote_code: bool = False
num_samples: Optional[int] = None
hf_dataset_split: Optional[str] = "test"
hf_token: Optional[str] = None
hf_dataset_version: Optional[str] = None
hf_dataset_input_column: Optional[str] = None
hf_dataset_reference_column: Optional[str] = None
hf_dataset_context_column: Optional[str] = None
use_ablation: bool = False
shuffle_input_samples: bool = False
all_agents_generate_first_draft: bool = False
all_agents_generate_draft: bool = False
voting_protocols_with_alterations: bool = False
calculate_persona_diversity: bool = False
challenge_final_results: bool = False
judge_intervention: Optional[str] = None
judge_metric: Optional[str] = None
judge_endpoint_url: Optional[str] = None
judge_api_key: str = "-"
judge_always_intervene: bool = False
```

### Discussion Parameters:
Response Generators: `freetext`, `simple`, `splitfreetext`

Decision Protocols: `approval_voting`, `consensus_voting`, `cumulative_voting`, `hybrid_consensus`, `majority_consensus`, `ranked_voting`, `simple_voting`, `summary`, `supermajority_consensus`, `unanimity_consensus`

Persona Generators: `expert`, `ipip`, `mock`, `nopersona`

Discussion Paradigms: `collective_refinement`, `debate`, `memory`, `relay`, `report`

## Evaluation

We provide some basic evaluation metrics that can be directly applied to the output json of mallm.
Supported metrics: `answerability`, `bertscore`, `bleu`, `ifeval`, `includes_answer`, `meteor`, `multichoice`, `rouge`, `squad`

From terminal:

`mallm-evaluate --input_json_file_path=test_out.json --output_json_file_path=test_out_evaluated.json --metrics=[bleu,rouge]`

From script:

```py
from mallm.evaluation.evaluator import Evaluator

evaluator = Evaluator(input_file_path="test_out.json", metrics=["bleu", "rouge"], extensive=False)
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
- The batch executor [script](https://github.com/Multi-Agent-LLMs/mallm/blob/main/mallm/scripts/batch_mallm.py) is located in the `mallm/scripts` folder and is named `batch_mallm.py`.
- A [template](https://github.com/Multi-Agent-LLMs/mallm/tree/main/mallm/scripts/batch.json.template) for the batch configuration file is provided as `batch.json.template` in the same folder.

### Setup

1. **Prepare your configuration file:**
   - Copy the `batch.json.template` file and rename it (e.g., `my_batch_config.json`).
   - Edit the JSON file to define your configurations. The file has four main sections:
     - `name`: A descriptive name for the batch of runs. This is optional but will be added to the output filename and can help identify the purpose of the batch.
     - `repeats`: The number of times to repeat each run. This is useful for running multiple trials with the same configuration.
     - `common`: Contains settings that apply to all runs unless overridden.
     - `runs`: An array of run-specific configurations.

   Example:
   ```json
   {
     "name": "test",
     "repeats": 2,
     "common": {
       "model_name": "gpt-3.5-turbo",
       "max_turns": 10,
       "num_agents": 3
     },
     "runs": [
       {
         "input_json_file_path": "path/to/data1.json",
         "output_json_file_path": "path/to/output1.json",
         "task_instruction_prompt": "Instruction for run 1"
       },
       {
         "input_json_file_path": "path/to/data2.json",
         "output_json_file_path": "path/to/output2.json",
         "task_instruction_prompt": "Instruction for run 2",
         "model_name": "gpt-4",
         "max_turns": 15
       }
     ]
   }
   ```

   In this example, the second run overrides the `model_name` and `max_turns` settings from the common configuration.

2. **Ensure all required dependencies are installed.**

### Running the Batch Executor

To run the batch executor, use the following command from the terminal:

`mallm-batch path/to/your/batch_config.json`

### Behavior

- The batch executor will process each run configuration in the order they appear in the JSON file.
- For each run:
  - It will create a `Config` object by merging the common settings with the run-specific settings.
  - It will then initialize a `Scheduler` with this configuration and run it.
  - Progress and any errors will be printed to the console.
- If a configuration is invalid or encounters an error during execution, the batch processor will skip to the next run.
- The process continues until all runs have been attempted.

### Tips

- Place settings that are common to most or all runs in the `common` section to reduce repetition.
- Run-specific settings will override common settings if both are specified.
- Always test your configurations individually before running them in a batch to ensure they work as expected.
- Use descriptive output file names to easily identify the results of each run.
- Monitor the console output for any error messages or skipped configurations.

By using the batch executor with common settings, you can easily manage multiple experiments or process various datasets with shared parameters, saving time and reducing the chance of configuration errors.

## Contributing
If you want to contribute, please use this pre-commit hook to ensure the same formatting for everyone.
```bash
pip install pre-commit
pre-commit install
```

### Testing
You can run unit tests locally:
`pytest ./test/`

## Citation
If you use this repository or the [paper](https://arxiv.org/abs/2410.22932) for your research work, please cite it in the following way.

```
@misc{becker2024multiagentlargelanguagemodels,
      title={Multi-Agent Large Language Models for Conversational Task-Solving}, 
      author={Jonas Becker},
      year={2024},
      eprint={2410.22932},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.22932}, 
}
```
