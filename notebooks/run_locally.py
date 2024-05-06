# This is a legacy file we can delete later if not needed

from tqdm import tqdm
from mallm.coordinator import *
import logging

# Configure logging for the library
library_logger = logging.getLogger("mallm")
library_logger.setLevel(logging.INFO)

# Add handlers to the logger
stream_handler = logging.StreamHandler()

# Optionally set a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

# Attach the handler to the logger
library_logger.addHandler(stream_handler)

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def create_llm_locally(ckpt_dir):
    """
    Initializes the LLM that the agents are using to generate their outputs.
    The LLM is set in evaluation mode. Thus, it immediately forgets everything that happened.
    It allows for an all-fresh reprompting at each iteration of the discussion.
    Any model within the huggingface format can be loaded.
    Returns HuggingFacePipeline
    """
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )
    model_config = transformers.AutoConfig.from_pretrained(ckpt_dir)
    if (
        device == "cpu"
    ):  # not recommended but useful for developing with no GPU available
        model = transformers.AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            config=model_config,
            offload_folder="offload",
            device_map="auto",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
    model.eval()
    print(f"Model {ckpt_dir} loaded on {device}")

    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_dir)
    # self.llm_tokenizer.pad_token_id = model.config.eos_token_id
    print("Using this tokenizer: " + str(llm_tokenizer.__class__.__name__))

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=llm_tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        pad_token_id=llm_tokenizer.eos_token_id,
        # model parameters
        do_sample=True,
        temperature=0.9,
        max_new_tokens=512,  # max number of tokens to generate in the output
        min_new_tokens=2,  # always answer something (no empty responses)
        repetition_penalty=1.1,  # without this output begins repeating
    )

    return HuggingFacePipeline(pipeline=pipeline)


def run_locally(
    d,
    out,
    instruction,
    ckpt_dir,
    use_moderator=False,
    max_turns=10,
    feedback_sentences=[3, 4],
    paradigm="memory",
    context_length=1,
    include_current_turn_in_memory=False,
    verbose=False,
):
    print("Running the model locally.")

    coordinator = Coordinator(
        use_moderator=use_moderator, llm=create_llm_locally(ckpt_dir), verbose=verbose
    )
    output_dicts = []

    for sample in tqdm(d):
        coordinator.cleanMemoryBucket()

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

        print(
            f"--> Agents discussed for {'%.2f' % discussionTime} seconds ({'%.2f' % (float(discussionTime) / 60.0)} minutes) to get the final answer: \n"
            + str(answer)
        )

        output_dicts.append(
            {
                "dataset": os.path.basename(data),
                "exampleId": sample["exampleId"],
                "datasetId": sample["datasetId"],
                "instruction": instruction,
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
        with open(out, "w") as file:
            file.write(json.dumps(output_dicts))
            file.truncate()


def main(
    data,
    out,
    instruction,
    use_moderator=False,
    max_turns=10,
    feedback_sentences=[3, 4],
    paradigm="memory",
    context_length=1,
    include_current_turn_in_memory=False,
    verbose=False,
    ckpt_dir=None,
):
    """
    The routine that starts the discussion between LLM agents iteratively on the provided data.
    """
    if not os.path.exists(data):
        print(
            "The input file you provided does not exist. Please specify a json lines file using --data."
        )
        return
    if not data.endswith(".json"):
        print(
            "The input file you provided is not a json file. Please specify a json lines file using --data."
        )
        return
    if not out.endswith(".json"):
        print(
            "The output file does not seem to be a json file. Please specify a file path using --out."
        )
        return

    # Cleaning other files
    if os.path.exists(out):
        os.remove(out)
        print(f"The file {out} has been deleted.")

    # Read input data (format: json lines)
    print(f"Reading {data}...")
    d = []
    with open(data) as f:
        for line in f:
            try:
                d.append(json.loads(line))
            except ValueError as e:
                print(
                    f"Invalid JSON in {data}! Please provide the input data in json lines format: {e}"
                )
    print(f"Found {len(d)} samples to discuss.")
    if ckpt_dir:
        run_locally(
            d,
            out,
            instruction,
            use_moderator,
            max_turns,
            feedback_sentences,
            paradigm,
            context_length,
            include_current_turn_in_memory,
            verbose,
        )
    else:
        print("No checkpoint directory was provided. No discussion is being held.")


if __name__ == "__main__":
    fire.Fire(main)
