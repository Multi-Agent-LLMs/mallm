from typing import Optional

from mallm.utils.types import TemplateFilling


def generate_chat_prompt_extract_result(result: str) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": "Extract the final solution to the task from the provided text. Remove statements of agreement, disagreement, and explanations. Do not modify the text.",
        },
        {
            "role": "user",
            "content": f"Text: {result}",
        },
    ]

    return prompts


def generate_chat_prompt_baseline(data: TemplateFilling) -> list[dict[str, str]]:
    # Use Zero-Shot-CoT: https://arxiv.org/pdf/2205.11916
    prompts = [
        {
            "role": "system",
            "content": "Solve the following task: {data.task_instruction} \nInput: {data.input_str}",
        },
        {
            "role": "assistant",
            "content": "Let's think step by step.",
        },
    ]
    return prompts
