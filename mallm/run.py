from tqdm import tqdm
from mallm.discourse_policy.coordinator import *
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

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def main(data, out, instruction, use_moderator=False, max_turns=10, feedback_sentences=[3, 4], paradigm="memory",
         context_length=1, include_current_turn_in_memory=False, verbose=False):
    '''
    The routine that starts the discussion between LLM agents iteratively on the provided data.
    '''

    if not os.path.exists(data):
        print("The input file you provided does not exist. Please specify a json lines file using --data.")
        return
    if not data.endswith(".json"):
        print("The input file you provided is not a json file. Please specify a json lines file using --data.")
        return
    if not out.endswith(".json"):
        print("The output file does not seem to be a json file. Please specify a file path using --out.")
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
                print(f"Invalid JSON in {data}! Please provide the input data in json lines format: {e}")
    print(f"Found {len(d)} samples to discuss.")

    coordinator = Coordinator(use_moderator=use_moderator, verbose=verbose)
    output_dicts = []

    for sample in tqdm(d):
        coordinator.cleanMemoryBucket()

        answer, globalMem, agentMems, turn, agreements, discussionTime = coordinator.discuss(
            instruction,
            sample["input"],
            sample["context"],
            use_moderator,
            feedback_sentences=feedback_sentences,
            paradigm=paradigm,
            max_turns=max_turns,
            context_length=context_length,
            include_current_turn_in_memory=include_current_turn_in_memory
        )

        print(
            f"--> Agents discussed for {'%.2f' % discussionTime} seconds ({'%.2f' % (float(discussionTime) / 60.0)} minutes) to get the final answer: \n" + str(
                answer))

        output_dicts.append({
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
            "clockSeconds": float('%.2f' % discussionTime),
            "globalMemory": globalMem,
            "agentMemory": agentMems
        })
        with open(out, 'w') as file:
            file.write(json.dumps(output_dicts))
            file.truncate()


if __name__ == "__main__":
    fire.Fire(main)
