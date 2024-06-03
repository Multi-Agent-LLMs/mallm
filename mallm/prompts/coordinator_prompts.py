from typing import Optional


def generate_chat_prompt_extract_result(
    instruction: Optional[str], result: str
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": "Extract the final solution to the task from the provided text. Remove statements of agreement, disagreement, and explanations. Do not modify the text.",
        }
    ]

    if instruction:
        user_content = f"\n\Task: {instruction}\n\nText: {result}"
    else:
        user_content = result

    prompts.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    return prompts


def generate_chat_prompt_baseline(
    task_instruction: str, input_str: str
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": "Please consider the example provided and think it step by step.",
        },
        {
            "role": "system",
            "content": f"Task: {task_instruction}",
        },
        {
            "role": "system",
            "content": f"Input: {input_str}",
        },
        {
            "role": "user",
            "content": "Utilize your talent and critical thinking to provide a solution.",
        },
    ]
    return prompts
