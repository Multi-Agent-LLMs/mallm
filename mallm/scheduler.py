import dataclasses
import gc
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from datetime import timedelta
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Optional

import fire
import httpx
import langchain
import langchain_core
import openai
from datasets import load_dataset
from openai import OpenAI
from rich import print
from rich.logging import RichHandler
from rich.progress import Console, Progress, TaskID  # type: ignore

from mallm.coordinator import Coordinator
from mallm.models.Chat import Chat
from mallm.utils.config import Config
from mallm.utils.dicts import RESPONSE_GENERATORS
from mallm.utils.types import InputExample, Response

FORMAT = "%(message)s"

logger = logging.getLogger("mallm")
logger.setLevel(logging.DEBUG)
handler = RichHandler(
    rich_tracebacks=True,
    tracebacks_suppress=[openai, httpx, langchain, langchain_core],
    markup=True,
)
formatter = logging.Formatter(fmt=FORMAT, datefmt="[%X]")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.basicConfig(filename="log.txt", filemode="w")


class Scheduler:
    def __init__(self, config: Config) -> None:
        config.check_config()

        # Cleaning other files
        if os.path.exists(config.out):
            os.remove(config.out)
            logger.info(f"""The file {config.out} has been deleted.""")

        # Read input data (format: json lines)
        try:
            logger.info(f"""Trying to read {config.data} from file...""")
            with open(config.data) as f:
                self.dataset_name = f.name
                json_data = json.loads(f.readline())
                self.data = [InputExample(**data) for data in json_data]
        except Exception as e:
            logger.warning(
                f"""Could not read {config.data} from file: {e}. Trying Hugging Face"""
            )

        if not self.data:
            try:
                # Read input data (format: huggingface dataset)
                logger.info(f"""Trying to read {config.data} from Hugging Face...""")
                self.dataset_name = config.data
                # Load from Hugging Face
                dataset = load_dataset(
                    self.dataset_name,
                    config.hf_dataset_version,
                    split=config.hf_dataset_split,
                    trust_remote_code=config.trust_remote_code,
                    token=config.hf_token,
                )
                # Put in native mallm format
                self.data = [
                    InputExample(
                        example_id=str(uuid.uuid4()),
                        dataset_id=None,
                        inputs=(
                            [x.pop(config.hf_dataset_input_column, None)]
                            if x.get(config.hf_dataset_input_column) is not None
                            else []
                        ),
                        context=(
                            [x.pop(config.hf_dataset_context_column, None)]
                            if x.get(config.hf_dataset_context_column) is not None
                            else None
                        ),
                        references=(
                            [x.pop(config.hf_dataset_reference_column, None)]
                            if x.get(config.hf_dataset_reference_column) is not None
                            else []
                        ),
                    )
                    for x in dataset
                ]

                # Filter if there are no inputs or references or they are empty
                self.data = [x for x in self.data if x.inputs and x.references]

            except Exception as e:
                logger.error(f"""Error reading {config.data} from Hugging Face: {e}""")
                sys.exit(1)

        try:
            for data in self.data:
                data.confirm_types()
        except AssertionError as e:
            logger.error(
                f"Input data has wrong format. Please delete and download the data again:\n{e}"
            )
            sys.exit(1)

        if config.shuffle_input_samples:
            random.shuffle(self.data)
            logger.info("Shuffled the input data.")

        self.config = config
        self.llm = Chat(
            client=OpenAI(
                base_url=self.config.endpoint_url, api_key=self.config.api_key
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
        self.ablation_output_dicts: list[dict[str, Any]] = []

        logger.info(f"""Found {self.total_samples} samples to process.""")

        logger.info("Finished initializing the scheduler.")

    def run_discussion(
        self,
        client: httpx.Client,
        sample: InputExample,
        console: Optional[Console],
        progress: Progress,
        task: TaskID,
    ) -> Optional[str]:
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
                console=console,
            )
        except Exception as e:
            logger.error("Failed intializing coordinator.")
            logger.error(e)
            self.failed_example_ids.append(sample.example_id)
            return None
        try:
            (
                answer,
                global_mem,
                agent_mems,
                turn,
                agreements,
                discussion_time,
                decision_success,
            ) = coordinator.discuss(config=self.config, sample=sample)
        except Exception:
            # More extensive error logging to ease debugging during async execution
            logger.error(f"Failed discussion of sample {sample.example_id}.")
            self.failed_example_ids.append(sample.example_id)
            logger.error("Exception occurred", exc_info=True)
            logger.error(traceback.format_exc())
            return None

        coordinator.setup_personas(
            self.config, input_lines=sample.inputs, context=sample.context
        )

        self.output_dicts.append(
            {
                "dataset": self.dataset_name,
                "exampleId": sample.example_id,
                "datasetId": sample.dataset_id,
                "instruction": self.config.instruction_prompt,
                "coordinatorId": coordinator.id,
                "personas": coordinator.get_agents(),
                "paradigm": self.config.paradigm,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer or None,
                "references": sample.references,
                "decisionSuccess": decision_success,
                "agreements": [
                    dataclasses.asdict(agreement) for agreement in agreements
                ],
                "turns": turn,
                "clockSeconds": float(f"{discussion_time:.2f}"),
                "globalMemory": [dataclasses.asdict(memory) for memory in global_mem],
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
        progress.update(task, advance=1)
        del coordinator
        gc.collect()

        if self.config.ablation:
            self.run_ablation(
                client, sample, len(self.output_dicts[-1]["globalMemory"])
            )

        return answer

    def manage_discussions(self, client: httpx.Client) -> None:
        """
        Manages all discussions on the data.
        Discussions are handled in a queue of length max_concurrent_requests.
        Once a spot in the queue is free because a discussion ended, the next discussion is initialized.
        """
        logger.debug("Starting discussion manager...")
        console = Console(record=True)
        # Creating HuggingFace endpoint

        if self.config.num_samples:
            processing_data = self.data[: self.config.num_samples]
            self.data = self.data[self.config.num_samples :]
        else:
            processing_data = self.data
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Finished discussions...", total=len(processing_data)
            )
            while True:
                logger.info(f"Processing {len(processing_data)} samples.")
                pool = ThreadPool(processes=self.config.max_concurrent_requests)
                results = []

                for sample in processing_data:
                    try:
                        results.append(
                            pool.apply_async(
                                self.run_discussion,
                                (client, sample, console, progress, task),
                            )
                        )
                    except Exception as e:
                        logger.error("Failed to run discussion.")
                        logger.error(e)
                pool.close()  # Done adding tasks.
                pool.join()  # Wait for all tasks to complete.
                del pool

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

    def run_ablation(
        self, client: httpx.Client, sample: InputExample, exchanged_messages: int
    ) -> Optional[str]:
        """
        Run an ablation where a single LM iteratively improves the solution the same number of times as the multi-agent system.
        """
        logger.info(
            f"""Starting ablation processing of sample {sample.example_id} with {exchanged_messages} iterative improvements."""
        )
        start_time = time.perf_counter()

        sample_instruction = self.config.instruction_prompt
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

        start_time = time.perf_counter()

        globalMemory = []
        answer = Response(
            message="None. Please provide a first solution.",
            solution="None. Please provide a first solution.",
            agreement=None,
        )
        for i in range(exchanged_messages):
            try:
                answer = self.response_generator.generate_ablation(
                    task_instruction=sample_instruction,
                    input_str=input_str,
                    current_solution=answer.solution,
                    chain_of_thought=self.config.chain_of_thought,
                )
                globalMemory.append(
                    {
                        "message_id": i,
                        "turn": i,
                        "message": answer.message,
                        "solution": answer.solution,
                    }
                )
            except Exception as e:
                logger.error("Failed running baseline.")
                logger.error(e)
                self.failed_example_ids.append(sample.example_id)
                return None

        discussion_time = timedelta(
            seconds=time.perf_counter() - start_time
        ).total_seconds()
        logger.info(
            f"""--> Baseline LM generated the final answer within {f'{discussion_time:.2f}'} seconds: \n"""
            + str(answer.solution)
        )

        self.ablation_output_dicts.append(
            {
                "dataset": self.dataset_name,
                "exampleId": sample.example_id,
                "datasetId": sample.dataset_id,
                "instruction": self.config.instruction_prompt,
                "coordinatorId": None,
                "personas": None,
                "paradigm": None,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer.solution or None,
                "references": sample.references,
                "decisionSuccess": None,
                "agreements": None,
                "turns": exchanged_messages,
                "clockSeconds": float(f"{discussion_time:.2f}"),
                "globalMemory": globalMemory,
                "agentMemory": None,
            }
        )
        try:
            out_path = Path(self.config.out)
            with open(
                out_path.with_name(out_path.stem + "-ablation.json"), "w"
            ) as file:
                file.write(
                    json.dumps(self.ablation_output_dicts)
                )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
                file.truncate()
        except Exception as e:
            logger.error("Failed to write ablation output to file.")
            logger.error(e)

        return answer.solution

    def run_baseline(
        self,
        client: httpx.Client,
        sample: InputExample,
    ) -> Optional[str]:
        """
        Task a single LM to solve a sample.
        """
        sample_instruction = self.config.instruction_prompt
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
                "instruction": self.config.instruction_prompt,
                "coordinatorId": None,
                "personas": None,
                "paradigm": None,
                "input": sample.inputs,
                "context": sample.context,
                "answer": answer.solution or None,
                "references": sample.references,
                "decisionSuccess": None,
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
            del pool

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

    def run(self) -> None:
        """
        The routine that runs the discussions between LLM agents on the provided data.
        """
        with httpx.Client() as client:
            if self.config.baseline:
                self.manage_baseline(client)  # baseline (single LM)
            else:
                self.manage_discussions(client)  # multi-agent discussion


def main() -> None:
    width = 70
    print("\n" + "=" * width)
    print("CONFIGURATION PARAMETERS".center(width))
    print("=" * width + "\n")
    config = fire.Fire(Config, serialize=print)
    print("\n" + "=" * width)
    print("END OF CONFIGURATION PARAMETERS".center(width))
    print("=" * width + "\n")
    scheduler = Scheduler(config)
    scheduler.run()
    print("Done.")


if __name__ == "__main__":
    main()
