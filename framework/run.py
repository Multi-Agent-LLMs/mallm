import os, sys
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
sys.path.append("/beegfs/wahle/github/MALLM")
import fire, glob
from tqdm import tqdm
from setup import *
from framework.discourse_policy.coordinator import *

def main(task_name, data, out, decision_threshold = None, use_moderator = True, max_turns = 10, feedback_length = 3, paradigm = "memory"):
    if not os.path.exists(data):
        print("The input file you provided does not exist. Please specify a json lines file using --data.")
    if not data.endswith(".json"):
        print("The input file you provided is not a json file. Please specify a json lines file using --data.")
    
    # Cleaning up the memory bucket
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
    print("Cleaned the memory bucket.")

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

    for sample in tqdm(d):
        answer, globalMem, agentMems = coordinator.discuss(
            task_name, 
            sample["task_instruction"], 
            sample["input"],
            decision_threshold, 
            use_moderator, 
            avg_feedback_length = feedback_length, 
            paradigm = paradigm, 
            max_turns = max_turns
        )
        print("--> Final answer: " + str(answer))

        with open(out, 'a') as file:
            file.write(json.dumps(
                {
                    "task_instruction": sample["task_instruction"], 
                    "input": sample["input"], 
                    "answer": answer,
                    "global_memory": globalMem,
                    "agent_memory": agentMems
                }))
            file.write('\n')

if __name__ == "__main__":
    fire.Fire(main)