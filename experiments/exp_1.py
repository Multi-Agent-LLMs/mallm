import os
import time
from datetime import timedelta
from pathlib import Path

import fire

from mallm import run_async
from mallm.config import *

dirname = os.path.dirname(__file__)


def main(
    data,
    instruction,
    hf_api_token,
    endpoint_url="http://127.0.0.1:8080/",
    use_moderator=False,
    max_turns=10,
    feedback_sentences=[3, 4],
    context_length=1,
    include_current_turn_in_memory=False,
    max_concurrent_requests=100,
):
    # Read input data (format: json lines)

    print("Starting experiment 1.")
    print("Paradigms: " + str(PARADIGMS))
    print("Dataset: " + data)
    print("--------------------------")

    Path("experiments/out/exp1").mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    for p in PARADIGMS:
        run_async.main(
            data=data,
            endpoint_url=endpoint_url,
            hf_api_token=hf_api_token,
            out=f"experiments/out/exp1/exp1_{data.split('/')[-1].split('.')[0]}_{p}.json",
            instruction=instruction,
            use_moderator=False,
            max_turns=10,
            feedback_sentences=[3, 4],
            paradigm=p,
            context_length=1,
            include_current_turn_in_memory=False,
            max_concurrent_requests=max_concurrent_requests,
        )
        passed_time = (
            "%.2f" % timedelta(seconds=time.perf_counter() - start_time).total_seconds()
        )
        print(
            f"""Finished paradigm {p}. Time passed since experiment start: {passed_time}s or {'%.2f' % (float(passed_time) / 60.0)} minutes or {'%.2f' % (float(passed_time) / 60.0 / 60.0)} hours"""
        )
        break  # for testing
    return


if __name__ == "__main__":
    fire.Fire(main)
