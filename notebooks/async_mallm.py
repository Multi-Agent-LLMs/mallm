# This is a test file used for speed comparison with TGI

import asyncio
import os
import random
import time

import httpx
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# Environment setup
TGI_SERVER_URL = "http://127.0.0.1:8080/"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "XXX")
MAX_CONCURRENT_REQUESTS = 100
MIN_ROUNDS = 15
MAX_ROUNDS = 30
NUM_DISCUSSIONS = 1000
GLOBAL_START_TIME = time.time()

# Set random seed for reproducibility
random.seed(1234)

# Semaphore to limit the number of concurrent requests
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Track how many requests have been handled
num_all_requests_handled = 0
num_total_requests = MIN_ROUNDS * NUM_DISCUSSIONS

# Creating HuggingFace endpoint
hf_llm = HuggingFaceEndpoint(
    endpoint_url=TGI_SERVER_URL,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_API_TOKEN,
    timeout=240.0,
)  # type: ignore

# Example prompt template
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="Translate the following English text into French: {prompt}",
)

# Creating LLMChain
llmchain = LLMChain(llm=hf_llm, prompt=prompt_template)


async def process_round(
    client,
    chain,
    discussion_id,
    round_number,
    total_rounds,
    context,
    output_file,
    queue,
    active_discussions,
):
    """Process a single round of a discussion."""
    global num_all_requests_handled

    input_prompt = {
        "prompt": f"{context} -- Round {round_number} of Discussion {discussion_id}"
    }
    async with sem:
        try:
            result = await chain.ainvoke(input=input_prompt, client=client)
        except Exception as e:
            print("Failed.")
            print(e)
        num_all_requests_handled += 1
    output = result.get("text", "")
    next_context = output + context

    # Ensure the context does not exceed ca 3000 tokens (approximately 15000 characters)
    max_tokens = 3000
    approx_chars_per_token = 5
    max_context_length = max_tokens * approx_chars_per_token
    if len(next_context) > max_context_length:
        next_context = next_context[-max_context_length:]

    # Write this round's result to file
    with open(output_file, "a") as f:
        f.write(f"Round {round_number}, Output: {output}\n")

    # Prepare the next round
    if round_number < total_rounds:
        # Enqueue next round of the same discussion
        await queue.put((discussion_id, round_number + 1, total_rounds, next_context))
    else:
        active_discussions.remove(discussion_id)

    print(
        f"Discussion {discussion_id}, Round {round_number}, Output len: {len(output)}"
    )
    print(f"Queue size: {queue.qsize()}")
    print(f"Num active discussions: {len(active_discussions)}")
    print(f"Num requests handled: {num_all_requests_handled}/{num_total_requests}")
    print(f"Total time elapsed: {time.time() - GLOBAL_START_TIME:.2f}s")
    print(
        f"Predicted time remaining: {(time.time() - GLOBAL_START_TIME) / num_all_requests_handled * (num_total_requests - num_all_requests_handled):.2f}s"
    )


async def manage_rounds(client, num_discussions, output_file, loop):
    queue = asyncio.Queue()
    active_discussions = set()
    for i in range(num_discussions):
        total_rounds = MIN_ROUNDS  # random.randint(MIN_ROUNDS, MAX_ROUNDS)
        active_discussions.add(i)
        await queue.put(
            (i, 1, total_rounds, "Initial context for Discussion {}".format(i))
        )

    while active_discussions:
        if not queue.empty():
            discussion_id, round_number, total_rounds, context = await queue.get()
            asyncio.run_coroutine_threadsafe(
                process_round(
                    client,
                    llmchain,
                    discussion_id,
                    round_number,
                    total_rounds,
                    context,
                    output_file,
                    queue,
                    active_discussions,
                ),
                loop,
            )
        await asyncio.sleep(0.01)  # Small delay to prevent tight loop


async def main(num_discussions):
    output_file = "discussion_outputs.txt"
    loop = asyncio.get_event_loop()
    async with httpx.AsyncClient() as client:
        await manage_rounds(client, num_discussions, output_file, loop)


if __name__ == "__main__":
    asyncio.run(main(NUM_DISCUSSIONS))
    print(f"Total execution time: {time.time() - GLOBAL_START_TIME:.2f}s")
    print(f"Total discussions processed: {NUM_DISCUSSIONS}")
    print(
        f"Average throughput per discussion: {(time.time() - GLOBAL_START_TIME) / NUM_DISCUSSIONS:.2f}s"
    )
