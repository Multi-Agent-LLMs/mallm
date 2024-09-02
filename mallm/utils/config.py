import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import requests

from mallm.utils.dicts import PROMPT_TEMPLATES

logger = logging.getLogger("mallm")


@dataclass
class Config:
    # DO NOT overwrite these values once assigned.
    input_json_file_path: str
    output_json_file_path: str
    task_instruction_prompt: str = ""
    task_instruction_prompt_template: Optional[str] = None
    endpoint_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-3.5-turbo"
    api_key: str = "-"
    num_neutral_agents: int = 0
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
    agent_generator: str = "expert"
    trust_remote_code: bool = False
    num_samples: Optional[int] = None
    hf_dataset_split: Optional[str] = "test"
    hf_token: Optional[str] = None
    hf_dataset_version: Optional[str] = None
    hf_dataset_input_column: Optional[str] = None
    hf_dataset_reference_column: Optional[str] = None
    hf_dataset_context_column: Optional[str] = None
    all_agents_drafting: bool = True 
    use_ablation: bool = False
    shuffle_input_samples: bool = False
    all_agents_generate_first_draft: bool = False

    def __post_init__(self) -> None:
        if (
            not self.task_instruction_prompt
            and self.task_instruction_prompt_template in PROMPT_TEMPLATES
        ):
            self.task_instruction_prompt = PROMPT_TEMPLATES[
                self.task_instruction_prompt_template
            ]

    def check_config(self) -> None:
        # TODO: make this more robust and conclusive. All arguments should be checked for validity, making the use of MALLM as fool-proof as possible.
        if not self.task_instruction_prompt:
            logger.error(
                "Please provide an instruction using the --instruction_prompt argument or a template using --instruction_prompt_template."
            )
            sys.exit(1)
        if os.path.isfile(self.input_json_file_path):
            if not self.input_json_file_path.endswith(".json"):
                logger.error("The dataset path does not seem to be a json file.")
                sys.exit(1)
        else:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.get(
                f"https://datasets-server.huggingface.co/is-valid?dataset={self.input_json_file_path}",
                headers=headers,
            )
            if not response.json()["preview"]:
                logger.error("The huggingface dataset cannot be loaded.")
                sys.exit(1)

        if not self.output_json_file_path.endswith(".json"):
            logger.error(
                "The output file does not seem to be a json file. Please specify a file path using --out."
            )
            sys.exit(1)

        if "api.openai.com" in self.endpoint_url and self.api_key == "-":
            logger.error(
                "When using the OpenAI API, you need to provide a key with the argument: --api_key=<your key>"
            )
            sys.exit(1)
        if self.num_neutral_agents >= self.num_agents and not self.skip_decision_making:
            logger.error(
                "You need at least one non-neutral agent to allow for decision-making. Set num_neutral_agents < num_agents."
            )
            sys.exit(1)
        if self.endpoint_url.endswith("/"):
            logger.warning("Removing trailing / from the endpoint url.")
            self.endpoint_url = self.endpoint_url[:-1]
        if not self.num_neutral_agents and not self.all_agents_drafting:
            logger.warning(
                "Setting feedback_only=True without a moderator does not make sense. No solutions will be drafted."
            )
        try:
            logger.info("Testing availability of the endpoint...")
            page = requests.head(self.endpoint_url.replace("/v1", ""))
            logger.info("Status: " + str(page.status_code))
            assert page.status_code == 200
        except Exception as e:
            logger.error("HTTP Error: Could not connect to the provided endpoint url.")
            logger.error(e)
            sys.exit(1)
        if self.concurrent_api_requests > 250:
            logger.warning(
                "max_concurrent_requests is very large. Please make sure the API endpoint you are using can handle that many simultaneous requests."
            )
