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
    data: str
    out: str
    instruction_prompt: str = ""
    instruction_prompt_template: Optional[str] = None
    endpoint_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    api_key: str = "-"
    use_moderator: bool = False
    max_turns: int = 10
    force_all_turns: bool = False
    feedback_sentences: Optional[tuple[int, int]] = None
    paradigm: str = "memory"
    response_generator: str = "simple"
    decision_protocol: str = "hybrid_consensus"
    context_length: int = 3
    include_current_turn_in_memory: bool = True
    extract_all_drafts: bool = True
    debate_rounds: int = 2
    max_concurrent_requests: int = 100
    baseline: bool = False
    chain_of_thought: bool = True
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
    feedback_only: bool = False
    ablation: bool = False
    shuffle_input_samples: bool = False

    def __post_init__(self):
        if (
            not self.instruction_prompt
            and self.instruction_prompt_template in PROMPT_TEMPLATES
        ):
            self.instruction_prompt = PROMPT_TEMPLATES[self.instruction_prompt_template]

    def check_config(self) -> None:
        # TODO: make this more robust and conclusive. All arguments should be checked for validity, making the use of MALLM as fool-proof as possible.
        if not self.instruction_prompt:
            logger.error(
                "Please provide an instruction using the --instruction_prompt argument or a template using --instruction_prompt_template."
            )
            sys.exit(1)
        if os.path.isfile(self.data):
            if not self.data.endswith(".json"):
                logger.error("The dataset path does not seem to be a json file.")
                sys.exit(1)
        else:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.get(
                f"https://datasets-server.huggingface.co/is-valid?dataset={self.data}",
                headers=headers,
            )
            if not response.json()["preview"]:
                logger.error("The huggingface dataset cannot be loaded.")
                sys.exit(1)

        if not self.out.endswith(".json"):
            logger.error(
                "The output file does not seem to be a json file. Please specify a file path using --out."
            )
            sys.exit(1)

        if "api.openai.com" in self.endpoint_url and self.api_key == "-":
            logger.error(
                "When using the OpenAI API, you need to provide a key with the argument: --api_key=<your key>"
            )
            sys.exit(1)
        if self.endpoint_url.endswith("/"):
            logger.warning("Removing trailing / from the endpoint url.")
            self.endpoint_url = self.endpoint_url[:-1]
        if not self.use_moderator and self.feedback_only:
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
        if self.max_concurrent_requests > 250:
            logger.error(
                "max_concurrent_requests is very large. Please make sure the API endpoint you are using can handle that many simultaneous requests."
            )
            sys.exit(1)
