import dataclasses
import glob
import json
import logging
import os
import sys
import time
from datetime import timedelta
from multiprocessing.pool import ThreadPool
from typing import Any, Optional

import fire
import httpx
import requests
from colorama import just_fix_windows_console
from openai import OpenAI

from mallm.coordinator import Coordinator
from mallm.models.Chat import Chat
from mallm.models.personas.ExpertGenerator import ExpertGenerator
from mallm.prompts.coordinator_prompts import (
    generate_chat_prompt_baseline,
    generate_chat_prompt_extract_result,
)
from mallm.utils.CustomFormatter import CustomFormatter
from mallm.utils.types import InputExample

just_fix_windows_console()

# Configure logging for the library
library_logger = logging.getLogger("mallm")
library_logger.setLevel(logging.DEBUG)

# Add handlers to the logger
stream_handler = logging.StreamHandler()

# Optionally set a formatter
stream_handler.setFormatter(CustomFormatter())

# Attach the handler to the logger
library_logger.addHandler(stream_handler)

logging.basicConfig(filename="log.txt", filemode="w")
logger = logging.getLogger("mallm")

output_dicts: list[dict[str, Any]] = []

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


class Scheduler:
    def __init__(
        self,
        data_file: str,
        out_file: str,
        instruction: str,
        endpoint_url: str = "https://api.openai.com",
        model: str = "gpt-3.5-turbo",
        # use "tgi" for Text Generation Inference by HuggingFace or one of these: https://platform.openai.com/docs/models
        api_key: str = "-",
        use_moderator: bool = False,
        max_turns: int = 10,
        force_all_turns: bool = False,
        feedback_sentences: Optional[tuple[int, int]] = None,
        paradigm: str = "memory",
        decision_protocol: str = "majority_consensus",
        context_length: int = 3,
        include_current_turn_in_memory: bool = True,
        extract_all_drafts: bool = True,
        debate_rounds: Optional[int] = None,
        max_concurrent_requests: int = 100,
        clear_memory_bucket: bool = True,
        memory_bucket_dir: str = "./mallm/utils/memory_bucket/",
        baseline: bool = False,
        chain_of_thought: bool = True,
        num_agents: int = 3,
        agent_generator: str = "expert",
    ) -> None:
        # Check for the correct aruments provided
        # TODO: make this more robust and conclusive. All arguments should be checked for validity, making the use of MALLM as fool-proof as possible.
        if not os.path.exists(data_file):
            logger.error(
                "The input file you provided does not exist. Please specify a json lines file using --data."
            )
            sys.exit(1)
        if not data_file.endswith(".json"):
            logger.error(
                "The input file you provided is not a json file. Please specify a json lines file using --data."
            )
            sys.exit(1)
        if not out_file.endswith(".json"):
            logger.error(
                "The output file does not seem to be a json file. Please specify a file path using --out."
            )
            sys.exit(1)
        if "api.openai.com" in endpoint_url and api_key == "-":
            logger.error(
                "When using the OpenAI API, you need to provide a key with the argument: --api_key=<your key>"
            )
            sys.exit(1)
        if endpoint_url.endswith("/"):
            logger.warning("Removing trailing / from the endpoint url.")
            endpoint_url = endpoint_url[:-1]
        try:
            logger.info("Testing availability of the endpoint...")
            page = requests.get(endpoint_url)
            logger.info("Status: " + str(page.status_code))
        except Exception as e:
            logger.error("HTTP Error: Could not connect to the provided endpoint url.")
            logger.error(e)
            sys.exit(1)

        if max_concurrent_requests > 500:
            logger.error(
                "max_concurrent_requests is too large. TGI can only handle about 500 requests. Please make sure to leave computing for other poeple too. Recommended: ~250."
            )
            sys.exit(1)

        # Cleaning other files
        if os.path.exists(out_file):
            os.remove(out_file)
            logger.info(f"""The file {out_file} has been deleted.""")

        # Cleaning the memory bucked from previous runs
        if clear_memory_bucket:
            self.clean_memory_bucket(memory_bucket_dir)

        # Read input data (format: json lines)
        logger.info(f"""Reading {data_file}...""")
        with open(data_file) as f:
            self.dataset_name = f.name
            json_data = json.loads(f.readline())

        self.data = [InputExample(**data) for data in json_data]
        try:
            for data in self.data:
                data.confirm_types()
        except AssertionError as e:
            logger.error(
                "Input data has wrong format. Please delete and download the data again."
            )
            sys.exit(1)

        self.out = out_file
        self.instruction = instruction
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key
        self.use_moderator = use_moderator
        self.max_turns = max_turns
        self.force_all_turns = force_all_turns
        self.feedback_sentences = feedback_sentences
        self.paradigm = paradigm
        self.decision_protocol = decision_protocol
        self.context_length = context_length
        self.include_current_turn_in_memory = include_current_turn_in_memory
        self.max_concurrent_requests = max_concurrent_requests
        self.extract_all_drafts = extract_all_drafts
        self.debate_rounds = debate_rounds
        self.clear_memory_bucket = clear_memory_bucket
        self.memory_bucket_dir = memory_bucket_dir
        self.total_samples = len(self.data)
        self.completed_samples = 0
        self.baseline = baseline
        self.chain_of_thought = chain_of_thought
        self.num_agents = num_agents
        self.agent_generator = agent_generator
        logger.info(f"""Found {self.total_samples} samples to process.""")

        logger.info("Finished initializing the scheduler.")

    def run_discussion(
        self,
        client: httpx.Client,
        llm: Chat,
        sample: InputExample,
    ) -> Optional[str]:
        """
        Runs a single discussion between agents on a sample.
        """

        logger.info(f"""Starting discussion of sample {sample.example_id}""")
        try:
            coordinator = Coordinator(
                use_moderator=self.use_moderator,
                model=llm,
                agent_generator=self.agent_generator,
                client=client,
                memory_bucket_dir=self.memory_bucket_dir,
            )
        except Exception as e:
            logger.error("Failed intializing coordinator.")
            logger.error(e)
            return None

        try:
            (
                answer,
                extracted_answer,
                global_mem,
                agent_mems,
                turn,
                agreements,
                discussion_time,
            ) = coordinator.discuss(
                self.instruction,
                sample.inputs,
                sample.context,
                self.use_moderator,
                feedback_sentences=self.feedback_sentences,
                paradigm=self.paradigm,
                decision_protocol=self.decision_protocol,
                max_turns=self.max_turns,
                force_all_turns=self.force_all_turns,
                context_length=self.context_length,
                include_current_turn_in_memory=self.include_current_turn_in_memory,
                extract_all_drafts=self.extract_all_drafts,
                debate_rounds=self.debate_rounds,
                chain_of_thought=self.chain_of_thought,
                num_agents=self.num_agents,
            )
        except Exception:
            # More extensive error logging to ease debugging during async execution
            logger.error("Failed discussion.")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(exc_type)
            logger.error(exc_obj)
            deep_tb = exc_tb
            while deep_tb and deep_tb.tb_next:
                deep_tb = deep_tb.tb_next
                fname = os.path.split(deep_tb.tb_frame.f_code.co_filename)[1]
                logger.error(
                    f"""-> at {fname}:{deep_tb.tb_lineno}, deeper function level error"""
                )
            return None

        logger.info(
            f"""--> Agents discussed for {turn} turns, {'%.2f' % discussion_time} seconds ({'%.2f' % (float(discussion_time) / 60.0)} minutes) to get the final answer: \n"""
            + str(extracted_answer)
        )
        logger.info(f"""Reference answer: {sample.references}""")

        output_dicts.append(
            {
                "dataset": self.dataset_name,
                "exampleId": sample.example_id,
                "datasetId": sample.dataset_id,
                "instruction": self.instruction,
                "coordinatorId": coordinator.id,
                "personas": coordinator.get_agents(),
                "paradigm": self.paradigm,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer,
                "extracted_answer": extracted_answer,
                "references": sample.references,
                "agreements": [
                    dataclasses.asdict(agreement) for agreement in agreements
                ],
                "turns": turn,
                "clockSeconds": float("%.2f" % discussion_time),
                "globalMemory": [dataclasses.asdict(memory) for memory in global_mem],
                "agentMemory": [
                    [dataclasses.asdict(memory) for memory in agent]
                    for agent in agent_mems
                    if agent
                ],
            }
        )
        try:
            with open(self.out, "w") as file:
                file.write(
                    json.dumps(output_dicts)
                )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
                file.truncate()
        except Exception as e:
            logger.error("Failed to write output to file.")
            logger.error(e)

        self.completed_samples += 1
        logger.info(
            f"""Completed samples: {self.completed_samples}. Samples left: {self.total_samples - self.completed_samples}."""
        )
        return answer

    def manage_discussions(self, client: httpx.Client) -> None:
        """
        Manages all discussions on the data.
        Discussions are handled in a queue of length max_concurrent_requests.
        Once a spot in the queue is free because a discussion ended, the next discussion is initialized.
        """
        logger.debug("Starting discussion manager...")
        # Creating HuggingFace endpoint
        llm = Chat(
            client=OpenAI(base_url=f"{self.endpoint_url}/v1", api_key=self.api_key),
            model=self.model,
        )

        pool = ThreadPool(processes=self.max_concurrent_requests)
        results = []
        for sample in self.data:
            try:
                results.append(
                    pool.apply_async(
                        self.run_discussion,
                        (client, llm, sample),
                    )
                )
            except Exception as e:
                logger.error("Failed to run discussion.")
                logger.error(e)
        pool.close()  # Done adding tasks.
        pool.join()  # Wait for all tasks to complete.

        for i, result in enumerate(results):
            if result.successful():
                logger.info("Process %s was successful." % i)
            else:
                logger.error("Process %s failed!" % i)

    def run_baseline(
        self,
        client: httpx.Client,
        llm: Chat,
        sample: InputExample,
    ) -> Optional[str]:
        """
        Task a single LM to solve a sample.
        """
        sample_instruction = self.instruction
        if sample.context:
            for c in sample.context:
                sample_instruction += "\n" + c
        input_str = ""
        for num, input_line in enumerate(sample.inputs):
            if len(sample.inputs) > 1:
                input_str += str(num + 1) + ") " + input_line + "\n"
            else:
                input_str = input_line

        logger.info(f"""Starting baseline processing of sample {sample.example_id}""")
        try:
            start_time = time.perf_counter()
            answer = llm.invoke(
                generate_chat_prompt_baseline(
                    task_instruction=sample_instruction,
                    input_str=input_str,
                    chain_of_thought=self.chain_of_thought,
                ),
                client=client,
            )
            discussion_time = timedelta(
                seconds=time.perf_counter() - start_time
            ).total_seconds()

            extracted_answer = None
            if self.extract_all_drafts:
                extracted_answer = llm.invoke(
                    generate_chat_prompt_extract_result(answer),
                    client=client,
                )
        except Exception as e:
            logger.error("Failed running baseline.")
            logger.error(e)
            return None

        logger.info(
            f"""--> Baseline LM generated the final answer within {'%.2f' % discussion_time} seconds: \n"""
            + str(answer)
        )

        output_dicts.append(
            {
                "dataset": self.dataset_name,
                "exampleId": sample.example_id,
                "datasetId": sample.dataset_id,
                "instruction": self.instruction,
                "coordinatorId": None,
                "personas": None,
                "paradigm": None,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer,
                "extracted_answer": extracted_answer,
                "references": sample.references,
                "agreements": None,
                "turns": None,
                "clockSeconds": float("%.2f" % discussion_time),
                "globalMemory": None,
                "agentMemory": None,
            }
        )
        try:
            with open(self.out, "w") as file:
                file.write(
                    json.dumps(output_dicts)
                )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
                file.truncate()
        except Exception as e:
            logger.error("Failed to write output to file.")
            logger.error(e)

        self.completed_samples += 1
        logger.info(
            f"""Completed samples: {self.completed_samples}. Samples left: {self.total_samples - self.completed_samples}."""
        )
        return answer

    def manage_baseline(self, client: httpx.Client) -> None:
        """
        Manages all samples of the data.
        The LM answers the query with a single request with no discussion being held.
        """
        logger.debug("Starting baseline manager...")
        # Creating HuggingFace endpoint
        llm = Chat(
            client=OpenAI(base_url=f"{self.endpoint_url}/v1", api_key=self.api_key),
            model=self.model,
        )

        pool = ThreadPool(processes=self.max_concurrent_requests)
        results = []
        for sample in self.data:
            try:
                results.append(
                    pool.apply_async(
                        self.run_baseline,
                        (client, llm, sample),
                    )
                )
            except Exception as e:
                logger.error("Failed running baseline.")
                logger.error(e)
        pool.close()  # Done adding tasks.
        pool.join()  # Wait for all tasks to complete.

    def clean_memory_bucket(self, memory_bucket_dir: Optional[str] = None) -> None:
        """
        Deletes all stored global memory
        """
        if not memory_bucket_dir:
            memory_bucket_dir = self.memory_bucket_dir

        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.bak"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dat"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dir"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(memory_bucket_dir, "*.json"))
        for f in filelist:
            os.remove(f)
        logger.info("Cleaned the memory bucket.")

    def run(self) -> None:
        """
        The routine that starts the discussions between LLM agents iteratively on the provided data.
        """
        with httpx.Client() as client:
            if self.baseline:
                self.manage_baseline(client)  # baseline (single LM)
            else:
                self.manage_discussions(client)  # multi-agent discussion


def main(
    data: str,
    out: str,
    instruction: str,
    endpoint_url: str = "https://api.openai.com",
    model: str = "gpt-3.5-turbo",
    # use "tgi" for Text Generation Inference by HuggingFace or one of these: https://platform.openai.com/docs/models
    api_key: str = "-",
    use_moderator: bool = False,
    max_turns: int = 10,
    force_all_turns: bool = False,
    feedback_sentences: Optional[tuple[int, int]] = None,
    paradigm: str = "memory",
    decision_protocol: str = "hybrid_consensus",
    context_length: int = 3,
    include_current_turn_in_memory: bool = True,
    extract_all_drafts: bool = True,
    debate_rounds: Optional[int] = None,
    max_concurrent_requests: int = 100,
    clear_memory_bucket: bool = True,
    memory_bucket_dir: str = "./mallm/utils/memory_bucket/",
    baseline: bool = False,
    chain_of_thought: bool = True,
    num_agents: int = 3,
    agent_generator: str = "expert",
) -> None:
    scheduler = Scheduler(
        data,
        out,
        instruction,
        endpoint_url,
        model=model,
        api_key=api_key,
        use_moderator=use_moderator,
        max_turns=max_turns,
        force_all_turns=force_all_turns,
        feedback_sentences=feedback_sentences,
        paradigm=paradigm,
        decision_protocol=decision_protocol,
        context_length=context_length,
        include_current_turn_in_memory=include_current_turn_in_memory,
        extract_all_drafts=extract_all_drafts,
        debate_rounds=debate_rounds,
        max_concurrent_requests=max_concurrent_requests,
        clear_memory_bucket=clear_memory_bucket,
        memory_bucket_dir=memory_bucket_dir,
        baseline=baseline,
        chain_of_thought=chain_of_thought,
        num_agents=num_agents,
        agent_generator=agent_generator,
    )
    scheduler.run()


if __name__ == "__main__":
    fire.Fire(main)
