import os, sys
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
sys.path.append("/beegfs/wahle/github/MALLM")
import fire, glob
from tqdm import tqdm
from setup import *
import time
from datetime import timedelta
from framework.discourse_policy.coordinator import *

def main(data, out, use_moderator = False, max_turns = 10, feedback_sentences = [3,4], paradigm = "memory", context_length = 1, include_current_turn_in_memory = False):
    if not os.path.exists(data):
        print("The input file you provided does not exist. Please specify a json lines file using --data.")
        return
    if not data.endswith(".json"):
        print("The input file you provided is not a json file. Please specify a json lines file using --data.")
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
    
    coordinator = Coordinator(use_moderator)
    output_dicts = []

    for sample in tqdm(d):
        coordinator.cleanMemoryBucket()
        start_time = time.perf_counter()

        answer, globalMem, agentMems, turn, agreements = coordinator.discuss(
            sample["task_instruction"], 
            sample["input"],
            use_moderator, 
            feedback_sentences = feedback_sentences, 
            paradigm = paradigm, 
            max_turns = max_turns,
            context_length = context_length,
            include_current_turn_in_memory=include_current_turn_in_memory
        )

        discussion_time = '%.2f' % timedelta(seconds=time.perf_counter() - start_time).total_seconds()
        print(f"--> Agents discussed for {discussion_time} seconds to get the final answer: \n" + str(answer))

        output_dicts.append({
            "task_instruction": sample["task_instruction"],
            "personas": coordinator.personas,
            "paradigm": paradigm,
            "input": sample["input"],
            "answer": answer,
            "agreements": agreements,
            "turns": turn,
            "time": discussion_time,
            "global_memory": globalMem,
            "agent_memory": agentMems
        })
        with open(out, 'w') as file:
            file.write(json.dumps(output_dicts))
            file.truncate()

if __name__ == "__main__":
    fire.Fire(main)