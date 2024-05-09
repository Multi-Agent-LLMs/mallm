import glob
import sys, httpx, requests
from multiprocessing.pool import ThreadPool

from colorama import just_fix_windows_console
from huggingface_hub import InferenceClient
from openai import OpenAI

from mallm.discourse_policy.coordinator import *
from mallm.models.HFTGIChat import HFTGIChat
from mallm.models.personas.TGIPersonaGenerator import (
    PersonaGenerator,
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


def run_discussion(
    client,
    llm,
    agent_generator,
    sample,
    out,
    instruction,
    use_moderator=False,
    max_turns=10,
    feedback_sentences=[3, 4],
    paradigm="memory",
    context_length=1,
    include_current_turn_in_memory=False,
    memory_bucket_dir="./mallm/utils/memory_bucket/",
):
    """
    Runs a single discussion between agents on a sample.
    """

    logger.info(f"""Starting discussion of sample {sample["exampleId"]}""")
    try:
        coordinator = Coordinator(
            use_moderator=use_moderator,
            model=llm,
            client=client,
            memory_bucket_dir=memory_bucket_dir,
            agent_generator=agent_generator,
        )
    except Exception as e:
        logger.error("Failed intializing coordinator.")
        print(e)

    try:
        answer, globalMem, agentMems, turn, agreements, discussionTime = (
            coordinator.discuss(
                instruction,
                sample["input"],
                sample["context"],
                use_moderator,
                feedback_sentences=feedback_sentences,
                paradigm=paradigm,
                max_turns=max_turns,
                context_length=context_length,
                include_current_turn_in_memory=include_current_turn_in_memory,
            )
        )
    except Exception as e:
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
            "instruction": instruction,
            "coordinatorId": coordinator.id,
            "personas": coordinator.getAgents(),
            "paradigm": paradigm,
            "input": sample["input"],
            "context": sample["context"],
            "answer": answer,
            "references": sample["references"],
            "agreements": agreements,
            "turns": turn,
            "clockSeconds": float("%.2f" % discussionTime),
            "globalMemory": globalMem,
            "agentMemory": agentMems,
        }
    )

    try:
        with open(out, "w") as file:
            file.write(
                json.dumps(output_dicts)
            )  # TODO: ensure correct json formatting (sometimes there is an invalid escape sequence warning)
            file.truncate()
    except Exception as e:
        logger.error("Failed to write output to file.")
        logger.error(e)


def manage_discussions(
    client,
    data,
    endpoint_url,
    hf_api_token,
    out,
    instruction,
    use_moderator,
    max_turns,
    feedback_sentences,
    paradigm,
    context_length,
    include_current_turn_in_memory,
    max_concurrent_requests=100,
    memory_bucket_dir="./mallm/utils/memory_bucket/",
):
    """
    Manages all discussions on the data.
    Discussions are handled in a queue of length max_concurrent_requests.
    Once a spot in the queue is free because a discussion ended, the next discussion is initialized.
    """
    # TODO: Add support for ChatGPT (OpenAI)
    # Creating HuggingFace endpoint
    llm_client_oai = OpenAI(base_url=f"{endpoint_url}/v1", api_key="-")

    llm = HFTGIChat(client=llm_client_oai)

    agent_generator = TGIPersonaGenerator(client=llm_client_oai)

    pool = ThreadPool(processes=max_concurrent_requests)
    results = []
    for sample in data:
        try:
            results.append(
                pool.apply_async(
                    run_discussion,
                    (
                        client,
                        llm,
                        agent_generator,
                        sample,
                        out,
                        instruction,
                        use_moderator,
                        max_turns,
                        feedback_sentences,
                        paradigm,
                        context_length,
                        include_current_turn_in_memory,
                        memory_bucket_dir,
                    ),
                )
            )
        except Exception as e:
            logger.error("Failed to run discussion.")
            logger.error(e)
    pool.close()  # Done adding tasks.
    pool.join()  # Wait for all tasks to complete.

    for i, result in enumerate(results):
        if result.successful():
            logger.info("Process %s was successful. Result is %s" % (i, result.get()))
        else:
            logger.error("Process %s failed!" % i)


def cleanMemoryBucket(memory_bucket_dir):
    """
    Deletes all stored global memory
    """
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


def main(
    data,
    out,
    instruction,
    endpoint_url,
    hf_api_token,
    use_moderator=False,
    max_turns=10,
    feedback_sentences=[3, 4],
    paradigm="memory",
    context_length=1,
    include_current_turn_in_memory=False,
    max_concurrent_requests=100,
    clear_memory_bucket=True,
    memory_bucket_dir="./mallm/utils/memory_bucket/",
):
    """
    The routine that starts the discussions between LLM agents iteratively on the provided data.
    """

    # Check for the correct aruments provided
    # TODO: make this more robust and conclusive. All arguments should be checked for validity, making the use of MALLM as fool-proof as possible.
    if not os.path.exists(data):
        logger.error(
            "The input file you provided does not exist. Please specify a json lines file using --data."
        )
        return
    if not data.endswith(".json"):
        logger.error(
            "The input file you provided is not a json file. Please specify a json lines file using --data."
        )
        return
    if not out.endswith(".json"):
        logger.error(
            "The output file does not seem to be a json file. Please specify a file path using --out."
        )
        return
    if max_concurrent_requests > 500:
        logger.error(
            "max_concurrent_requests is too large. TGI can only handle about 500 requests. Please make sure to leave computing for other poeple too. Recommended: ~250."
        )
        return
    try:
        logger.info("Testing availability of the endpoint...")
        page = requests.get(endpoint_url)
        logger.info("Status: " + str(page.status_code))
    except Exception as e:
        logger.error("HTTP Error: Could not connect to the provided endpoint url.")
        logger.error(e)
        return

    # Cleaning other files
    if os.path.exists(out):
        os.remove(out)
        logger.info(f"""The file {out} has been deleted.""")

    # Cleaning the memory bucked from previous runs
    if clear_memory_bucket:
        cleanMemoryBucket(memory_bucket_dir)

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
    logger.info(f"""Found {len(d)} samples to discuss.""")

    with httpx.Client() as client:
        manage_discussions(
            client,
            d,
            endpoint_url,
            hf_api_token,
            out,
            instruction,
            use_moderator,
            max_turns,
            feedback_sentences,
            paradigm,
            context_length,
            include_current_turn_in_memory,
            max_concurrent_requests,
            memory_bucket_dir,
        )


if __name__ == "__main__":
    fire.Fire(main)
