from datasets import load_from_disk
import sys, os
sys.path.append("/beegfs/wahle/github/MALLM")
from setup import *
from tqdm import tqdm
import time
import fire, json
from datetime import timedelta
import framework.run
from pathlib import Path

dirname = os.path.dirname(__file__)

def main(data, instruction, use_moderator = False, max_turns = 10, feedback_sentences = [3,4], context_length = 1, include_current_turn_in_memory = False):
    # Read input data (format: json lines)

    print("Starting experiment 1.")
    print("Paradigms: " + str(PARADIGMS))
    print("Dataset: " + data)
    print("--------------------------")

    Path("experiments/out/exp1").mkdir(parents=True, exist_ok=True)
    
    start_time = time.perf_counter()
    for p in PARADIGMS:
        framework.run.main(data=data, out=f"experiments/out/exp1/exp1_{data.split('/')[-1].split('.')[0]}_{p}.json", instruction=instruction, use_moderator = False, max_turns = 10, feedback_sentences = [3,4], paradigm = "memory", context_length = 1, include_current_turn_in_memory = False)
        passed_time = '%.2f' % timedelta(seconds=time.perf_counter() - start_time).total_seconds()
        print(f"Finished {p}. Time passed since experiment start: {passed_time}s or {'%.2f' % (float(passed_time)/60.0)} minutes or {'%.2f' % (float(passed_time)/60.0/60.0)} hours")
    return

if __name__ == "__main__":
    fire.Fire(main)