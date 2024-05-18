# This is a test file used for speed comparison with TGI

import os
import random
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from multiprocessing.pool import ThreadPool
import time
import httpx
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

# Track how many requests have been handled
num_all_requests_handled = 0
num_total_requests = MIN_ROUNDS * NUM_DISCUSSIONS

# Creating HuggingFace endpoint
hf_llm = HuggingFaceEndpoint(
    endpoint_url=TGI_SERVER_URL,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_API_TOKEN,
)  # type: ignore

# Example prompt template
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="Translate the following English text into French: {prompt}",
)

# Creating LLMChain
llmchain = LLMChain(llm=hf_llm, prompt=prompt_template)

# Set random seed for reproducibility
random.seed(1234)


def run_discussion(client, chain, output_file, rounds):
    global num_all_requests_handled

    context = "Initial context for Discussion"
    print(f"Starting discussion with {rounds} rounds.")
    discussion_results = []
    for i in range(rounds):
        input_prompt = {"prompt": f"{context} -- Round {i+1} of Discussion"}
        start_time = time.time()
        try:
            result = chain.invoke(input=input_prompt, client=client)
        except Exception as e:
            print("Failed.")
            print(e)
        print(result)
        num_all_requests_handled += 1
        duration = time.time() - start_time
        output = result["text"]
        discussion_results.append(
            f"Round {i+1}, Time: {duration:.2f}s, Output: {output}"
        )
        context = output + context
        print(f"Round {i+1}, Time: {duration:.2f}s, Output len: {len(output)}")

        # Ensure the context does not exceed ca 3000 tokens (approximately 15000 characters)
        max_tokens = 3000
        approx_chars_per_token = 5
        max_context_length = max_tokens * approx_chars_per_token
        if len(context) > max_context_length:
            context[-max_context_length:]

        # Write this round's result to file
        with open(output_file, "a") as f:
            f.write(f"Round {i}, Output: {output}\n")

        print(f"Num requests handled: {num_all_requests_handled}/{num_total_requests}")
        print(f"Total time elapsed: {time.time() - GLOBAL_START_TIME:.2f}s")
        print(
            f"Predicted time remaining: {(time.time() - GLOBAL_START_TIME) / num_all_requests_handled * (num_total_requests - num_all_requests_handled):.2f}s"
        )


def manage_discussions(
    client,
    num_discussions,
    output_file,
    max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
):
    pool = ThreadPool(processes=max_concurrent_requests)
    results = []
    for _ in range(num_discussions):
        rounds = MIN_ROUNDS  # random.randint(MIN_ROUNDS, MAX_ROUNDS)
        results.append(
            pool.apply_async(run_discussion, (client, llmchain, output_file, rounds))
        )

    pool.close()  # Done adding tasks.
    pool.join()  # Wait for all tasks to complete.

    for i, result in enumerate(results):
        if result.successful():
            print("Process %s was successful. Result is %s" % (i, result.get()))
        else:
            print("Process %s failed!" % i)


def main(num_discussions):
    output_file = "discussion_outputs.txt"
    with httpx.Client() as client:
        manage_discussions(client, num_discussions, output_file)


if __name__ == "__main__":
    start_main_time = time.time()
    main(NUM_DISCUSSIONS)
    print(f"Total execution time: {time.time() - start_main_time:.2f}s")
    print(f"Total discussions processed: {NUM_DISCUSSIONS}")
    print(
        f"Average throughput per discussion: {(time.time() - start_main_time) / NUM_DISCUSSIONS:.2f}s"
    )
