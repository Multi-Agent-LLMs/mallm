import dataclasses
import glob
import json
import logging
from typing import Optional
import os
import sys
from multiprocessing.pool import ThreadPool

import fire
import httpx
import requests
from colorama import just_fix_windows_console
from openai import OpenAI

from mallm.coordinator import Coordinator
from mallm.models.HFTGIChat import HFTGIChat
from mallm.models.personas.TGIPersonaGenerator import (
    TGIPersonaGenerator,
)
from mallm.utils.CustomFormatter import CustomFormatter

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

output_dicts = []

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


class Scheduler:
    def __init__(
        self,
        data: list[dict],
        out: str,
        instruction: str,
        endpoint_url: str,
        use_moderator: bool = False,
        max_turns: int = 10,
        feedback_sentences: tuple[int, int] = (3, 4),
        paradigm: str = "memory",
        decision_protocol: str = "majority_consensus",
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        extract_all_drafts: bool = False,
        debate_rounds=Optional[int],
        max_concurrent_requests: int = 100,
        clear_memory_bucket: bool = True,
        memory_bucket_dir: str = "./mallm/utils/memory_bucket/",
    ):
        # Check for the correct aruments provided
        # TODO: make this more robust and conclusive. All arguments should be checked for validity, making the use of MALLM as fool-proof as possible.
        if not os.path.exists(data):
            logger.error(
                "The input file you provided does not exist. Please specify a json lines file using --data."
            )
            sys.exit(1)
        if not data.endswith(".json"):
            logger.error(
                "The input file you provided is not a json file. Please specify a json lines file using --data."
            )
            sys.exit(1)
        if not out.endswith(".json"):
            logger.error(
                "The output file does not seem to be a json file. Please specify a file path using --out."
            )
            sys.exit(1)
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
        if os.path.exists(out):
            os.remove(out)
            logger.info(f"""The file {out} has been deleted.""")

        # Cleaning the memory bucked from previous runs
        if clear_memory_bucket:
            self.cleanMemoryBucket(memory_bucket_dir)

        # Read input data (format: json lines)
        logger.info(f"""Reading {data}...""")
        d = []
        with open(data) as f:
            for line in f:
                try:
                    d.append(json.loads(line))
                except ValueError as e:
                    logger.error(
                        f"""Invalid JSON in {data}! Please provide the input data in json lines format: {e}"""
                    )

        self.data = d
        self.out = out
        self.instruction = instruction
        self.endpoint_url = endpoint_url
        self.use_moderator = use_moderator
        self.max_turns = max_turns
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
        logger.info(f"""Found {self.total_samples} samples to discuss.""")

        logger.info("Finished initializing the scheduler.")

    def run_discussion(
        self,
        client: OpenAI,
        llm: HFTGIChat,
        agent_generator: TGIPersonaGenerator,
        sample: dict,
    ):
        """
        Runs a single discussion between agents on a sample.
        """

        logger.info(f"""Starting discussion of sample {sample["exampleId"]}""")
        try:
            coordinator = Coordinator(
                use_moderator=self.use_moderator,
                model=llm,
                agent_generator=agent_generator,
                client=client,
                memory_bucket_dir=self.memory_bucket_dir,
            )
        except Exception as e:
            logger.error("Failed intializing coordinator.")
            logger.error(e)

        try:
            answer, globalMem, agentMems, turn, agreements, discussionTime = (
                coordinator.discuss(
                    self.instruction,
                    sample["input"],
                    sample["context"],
                    self.use_moderator,
                    feedback_sentences=self.feedback_sentences,
                    paradigm=self.paradigm,
                    decision_protocol=self.decision_protocol,
                    max_turns=self.max_turns,
                    context_length=self.context_length,
                    include_current_turn_in_memory=self.include_current_turn_in_memory,
                    extract_all_drafts=self.extract_all_drafts,
                    debate_rounds=self.debate_rounds,
                )
            )
        except Exception:
            # More extensive error logging to ease debugging during async execution
            logger.error("Failed discussion.")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logger.error(exc_type)
            logger.error(exc_obj)
            deep_tb = exc_tb
            while deep_tb.tb_next:
                deep_tb = deep_tb.tb_next
                fname = os.path.split(deep_tb.tb_frame.f_code.co_filename)[1]
                logger.error(
                    f"""-> at {fname}:{deep_tb.tb_lineno}, deeper function level error"""
                )

        logger.info(
            f"""--> Agents discussed for {turn} turns, {'%.2f' % discussionTime} seconds ({'%.2f' % (float(discussionTime) / 60.0)} minutes) to get the final answer: \n"""
            + str(answer)
        )

        output_dicts.append(
            {
                "dataset": "placeholder",
                "exampleId": sample["exampleId"],
                "datasetId": sample["datasetId"],
                "instruction": self.instruction,
                "coordinatorId": coordinator.id,
                "personas": coordinator.get_agents(),
                "paradigm": self.paradigm,
                "input": sample["input"],
                "context": sample["context"],
                "answer": answer,
                "references": sample["references"],
                "agreements": [
                    dataclasses.asdict(agreement) for agreement in agreements
                ],
                "turns": turn,
                "clockSeconds": float("%.2f" % discussionTime),
                "globalMemory": globalMem,
                "agentMemory": agentMems,
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
            f"""Completed samples: {self.completed_samples}. Samples left: {self.total_samples-self.completed_samples}."""
        )
        return answer

    def manage_discussions(self, client: httpx.Client):
        """
        Manages all discussions on the data.
        Discussions are handled in a queue of length max_concurrent_requests.
        Once a spot in the queue is free because a discussion ended, the next discussion is initialized.
        """
        # Creating HuggingFace endpoint
        llm_client_oai = OpenAI(base_url=f"{self.endpoint_url}/v1", api_key="-")

        llm = HFTGIChat(client=llm_client_oai)

        agent_generator = TGIPersonaGenerator(client=llm_client_oai)

        pool = ThreadPool(processes=self.max_concurrent_requests)
        results = []
        for sample in self.data:
            try:
                results.append(
                    pool.apply_async(
                        self.run_discussion,
                        (client, llm, agent_generator, sample),
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

    def cleanMemoryBucket(self, memory_bucket_dir=None):
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

    def run(self):
        """
        The routine that starts the discussions between LLM agents iteratively on the provided data.
        """

        with httpx.Client() as client:
            self.manage_discussions(client)


def main(
    data: list[dict],
    out: str,
    instruction: str,
    endpoint_url: str,
    use_moderator: bool = False,
    max_turns: int = 10,
    feedback_sentences: tuple[int, int] = (3, 4),
    paradigm: str = "memory",
    decision_protocol: str = "majority_consensus",
    context_length: int = 1,
    include_current_turn_in_memory: bool = False,
    extract_all_drafts: bool = False,
    debate_rounds: Optional[int] = None,
    max_concurrent_requests: int = 100,
    clear_memory_bucket: bool = True,
    memory_bucket_dir: str = "./mallm/utils/memory_bucket/",
):
    scheduler = Scheduler(
        data,
        out,
        instruction,
        endpoint_url,
        use_moderator=use_moderator,
        max_turns=max_turns,
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
    )
    scheduler.run()


if __name__ == "__main__":
    fire.Fire(main)
