from typing import Optional

from mallm.utils.types import TemplateFilling


def generate_chat_prompt_extract_result(result: Optional[str]) -> list[dict[str, str]]:
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


def generate_chat_prompt_baseline(
    task_instruction: str, input_str: str, chain_of_thought: bool
) -> list[dict[str, str]]:
    # Use Zero-Shot-CoT: https://arxiv.org/pdf/2205.11916
    prompts = [
        {
            "role": "system",
            "content": "Solve the following task: {task_instruction} \nInput: {input_str}",
        }
    ]
    if chain_of_thought:
        prompts.append(
            {
                "role": "assistant",
                "content": "Let's think step by step.",
            }
        )
    return prompts
