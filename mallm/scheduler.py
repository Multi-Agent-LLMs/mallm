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
from colorama import just_fix_windows_console
from openai import OpenAI

from mallm.coordinator import Coordinator
from mallm.models.Chat import Chat
from mallm.utils.config import Config
from mallm.utils.CustomFormatter import CustomFormatter
from mallm.utils.dicts import RESPONSE_GENERATORS
from mallm.utils.types import InputExample
from mallm.utils.utils import pretty_print_dict, suppress_output

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


class Scheduler:
    def __init__(self, config: Config) -> None:
        config.check_config()

        # Cleaning other files
        if os.path.exists(config.out):
            os.remove(config.out)
            logger.info(f"""The file {config.out} has been deleted.""")

        # Cleaning the memory bucked from previous runs
        if config.clear_memory_bucket:
            self.clean_memory_bucket(config.memory_bucket_dir)

        # Read input data (format: json lines)
        logger.info(f"""Reading {config.data}...""")
        with open(config.data) as f:
            self.dataset_name = f.name
            json_data = json.loads(f.readline())

        self.data = [InputExample(**data) for data in json_data]
        try:
            for data in self.data:
                data.confirm_types()
        except AssertionError:
            logger.error(
                "Input data has wrong format. Please delete and download the data again."
            )
            sys.exit(1)

        self.config = config
        self.llm = Chat(
            client=OpenAI(
                base_url=f"{self.config.endpoint_url}/v1", api_key=self.config.api_key
            ),
            model=self.config.model,
        )

        if config.response_generator not in RESPONSE_GENERATORS:
            logger.error(f"No valid response generator for {config.response_generator}")
            raise Exception(
                f"No valid response generator for {config.response_generator}"
            )
        self.response_generator = RESPONSE_GENERATORS[config.response_generator](
            self.llm
        )

        self.completed_samples = 0
        self.total_samples = len(self.data)
        self.failed_example_ids: list[str] = []
        self.output_dicts: list[dict[str, Any]] = []

        logger.info(f"""Found {self.total_samples} samples to process.""")

        logger.info("Finished initializing the scheduler.")

    def run_discussion(
        self,
        client: httpx.Client,
        sample: InputExample,
    ) -> Optional[list[str]]:
        """
        Runs a single discussion between agents on a sample.
        """

        logger.info(f"""Starting discussion of sample {sample.example_id}""")
        try:
            coordinator = Coordinator(
                use_moderator=self.config.use_moderator,
                model=self.llm,
                agent_generator=self.config.agent_generator,
                client=client,
                memory_bucket_dir=self.config.memory_bucket_dir,
            )
        except Exception as e:
            logger.error("Failed intializing coordinator.")
            logger.error(e)
            return None

        coordinator.setup_personas(
            self.config, input_lines=sample.inputs, context=sample.context
        )

        answers = []

        for _ in range(10):
            coordinator.clear_global_memory()
            try:
                (
                    answer,
                    global_mem,
                    agent_mems,
                    turn,
                    agreements,
                    discussion_time,
                    decision_success,
                ) = coordinator.discuss(config=self.config)
            except Exception:
                # More extensive error logging to ease debugging during async execution
                logger.error(f"Failed discussion of sample {sample.example_id}.")
                self.failed_example_ids.append(sample.example_id)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logger.error(exc_type)
                logger.error(exc_obj)
                deep_tb = exc_tb
                while deep_tb and deep_tb.tb_next:
                    deep_tb = deep_tb.tb_next
                    f_name = os.path.split(deep_tb.tb_frame.f_code.co_filename)[1]
                    logger.error(
                        f"""-> at {f_name}:{deep_tb.tb_lineno}, deeper function level error"""
                    )
                return None

            logger.info(
                f"""--> Agents discussed for {turn} turns, {f'{discussion_time:.2f}'} seconds ({'%.2f' % (float(discussion_time) / 60.0)} minutes) to get the final answer: \n"""
                + str(answer)
            )
            logger.info(f"""Reference answer: {sample.references}""")
            logger.info(f"""Decision successful: {decision_success}""")

            self.output_dicts.append(
                {
                    "dataset": self.dataset_name,
                    "exampleId": sample.example_id,
                    "datasetId": sample.dataset_id,
                    "instruction": self.config.instruction,
                    "coordinatorId": coordinator.id,
                    "personas": coordinator.get_agents(),
                    "paradigm": self.config.paradigm,
                    "input": sample.inputs,
                    "context": sample.context,
                    "answer": answer or None,
                    "references": sample.references,
                    "decision_success": decision_success,
                    "agreements": [
                        dataclasses.asdict(agreement) for agreement in agreements
                    ],
                    "turns": turn,
                    "clockSeconds": float(f"{discussion_time:.2f}"),
                    "globalMemory": [
                        dataclasses.asdict(memory) for memory in global_mem
                    ],
                    "agentMemory": [
                        [dataclasses.asdict(memory) for memory in agent]
                        for agent in agent_mems
                        if agent
                    ],
                }
            )
            try:
                with open(self.config.out, "w") as file:
                    file.write(
                        json.dumps(self.output_dicts)
                    )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
                    file.truncate()
            except Exception as e:
                logger.error("Failed to write output to file.")
                logger.error(e)
                self.failed_example_ids.append(sample.example_id)

            self.completed_samples += 1
            logger.info(
                f"""Completed samples: {self.completed_samples}. Samples left: {self.total_samples - self.completed_samples}."""
            )

            if answer:
                answers.append(answer)
        return answers

    def manage_discussions(self, client: httpx.Client) -> None:
        """
        Manages all discussions on the data.
        Discussions are handled in a queue of length max_concurrent_requests.
        Once a spot in the queue is free because a discussion ended, the next discussion is initialized.
        """
        logger.debug("Starting discussion manager...")
        # Creating HuggingFace endpoint

        if self.config.num_samples:
            processing_data = self.data[: self.config.num_samples]
            self.data = self.data[self.config.num_samples :]
        else:
            processing_data = self.data

        while True:
            logger.info(f"Processing {len(processing_data)} samples.")
            pool = ThreadPool(processes=self.config.max_concurrent_requests)
            results = []
            for sample in processing_data:
                try:
                    results.append(
                        pool.apply_async(
                            self.run_discussion,
                            (client, sample),
                        )
                    )
                except Exception as e:
                    logger.error("Failed to run discussion.")
                    logger.error(e)
            pool.close()  # Done adding tasks.
            pool.join()  # Wait for all tasks to complete.

            if len(self.failed_example_ids) == 0:
                logger.info("No samples failed.")
                break
            logger.warning(
                f"{len(self.failed_example_ids)} samples failed. Here is a list of their example_ids: \n{self.failed_example_ids!s}"
            )
            if len(self.data) < len(self.failed_example_ids):
                logger.error(
                    "No more samples in the datasets to substitute failed samples."
                )
                raise Exception(
                    "No more samples in the datasets to substitute failed samples."
                )
            logger.warning("Resampling from the dataset as a substitute...")
            processing_data = self.data[: len(self.failed_example_ids)]
            self.data = self.data[len(self.failed_example_ids) :]
            self.failed_example_ids = []

    def run_baseline(
        self,
        client: httpx.Client,
        sample: InputExample,
    ) -> Optional[str]:
        """
        Task a single LM to solve a sample.
        """
        sample_instruction = self.config.instruction
        if sample.context:
            sample_instruction += "\nContext:"
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
            answer = self.response_generator.generate_baseline(
                task_instruction=sample_instruction,
                input_str=input_str,
                chain_of_thought=self.config.chain_of_thought,
            )
            discussion_time = timedelta(
                seconds=time.perf_counter() - start_time
            ).total_seconds()
        except Exception as e:
            logger.error("Failed running baseline.")
            logger.error(e)
            self.failed_example_ids.append(sample.example_id)
            return None

        logger.info(
            f"""--> Baseline LM generated the final answer within {f'{discussion_time:.2f}'} seconds: \n"""
            + str(answer.solution)
        )

        self.output_dicts.append(
            {
                "dataset": self.dataset_name,
                "exampleId": sample.example_id,
                "datasetId": sample.dataset_id,
                "instruction": self.config.instruction,
                "coordinatorId": None,
                "personas": None,
                "paradigm": None,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer.solution or None,
                "references": sample.references,
                "decision_success": None,
                "agreements": None,
                "turns": None,
                "clockSeconds": float(f"{discussion_time:.2f}"),
                "globalMemory": None,
                "agentMemory": None,
            }
        )
        try:
            with open(self.config.out, "w") as file:
                file.write(
                    json.dumps(self.output_dicts)
                )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
                file.truncate()
        except Exception as e:
            logger.error("Failed to write output to file.")
            logger.error(e)
            self.failed_example_ids.append(sample.example_id)

        self.completed_samples += 1
        logger.info(
            f"""Completed samples: {self.completed_samples}. Samples left: {self.total_samples - self.completed_samples}."""
        )
        return answer.solution

    def manage_baseline(self, client: httpx.Client) -> None:
        """
        Manages all samples of the data.
        The LM answers the query with a single request with no discussion being held.
        """
        logger.debug("Starting baseline manager...")
        # Creating HuggingFace endpoint

        if self.config.num_samples:
            processing_data = self.data[: self.config.num_samples]
            self.data = self.data[self.config.num_samples :]
        else:
            processing_data = self.data

        while True:
            logger.info(f"Processing {len(processing_data)} samples.")
            pool = ThreadPool(processes=self.config.max_concurrent_requests)
            results = []
            for sample in processing_data:
                try:
                    results.append(
                        pool.apply_async(
                            self.run_baseline,
                            (client, sample),
                        )
                    )
                except Exception as e:
                    logger.error("Failed running baseline.")
                    logger.error(e)
            pool.close()  # Done adding tasks.
            pool.join()  # Wait for all tasks to complete.

            if len(self.failed_example_ids) == 0:
                logger.info("No samples failed.")
                break
            logger.warning(
                f"{len(self.failed_example_ids)} samples failed. Here is a list of their example_ids: \n{self.failed_example_ids!s}"
            )
            if len(self.data) < len(self.failed_example_ids):
                logger.error(
                    "No more samples in the datasets to substitute failed samples."
                )
                raise Exception(
                    "No more samples in the datasets to substitute failed samples."
                )
            logger.warning("Resampling from the dataset as a substitute...")
            processing_data = self.data[: len(self.failed_example_ids)]
            self.data = self.data[len(self.failed_example_ids) :]
            self.failed_example_ids = []

    def clean_memory_bucket(self, memory_bucket_dir: Optional[str] = None) -> None:
        """
        Deletes all stored global memory
        """
        if not memory_bucket_dir:
            memory_bucket_dir = self.config.memory_bucket_dir

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
        logger.info(f"Cleaned the memory bucket {memory_bucket_dir!s}.")

    def run(self) -> None:
        """
        The routine that starts the discussions between LLM agents iteratively on the provided data.
        """
        with httpx.Client() as client:
            if self.config.baseline:
                self.manage_baseline(client)  # baseline (single LM)
            else:
                self.manage_discussions(client)  # multi-agent discussion


def main() -> None:
    with suppress_output():
        config = fire.Fire(Config, serialize=print)
    pretty_print_dict(config)
    scheduler = Scheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
