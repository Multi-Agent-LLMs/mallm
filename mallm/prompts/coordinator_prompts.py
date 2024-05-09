from langchain_core.prompts import ChatPromptTemplate


def generate_chat_prompt_extract_result(result):
    prompts = [
        {
            "role": "system",
            "content": "Extract the final result from the provided text. Do not output any additional text and remove the explanation. Only copy the result from the provided text without modifications.",
        },
        {
            "role": "user",
            "content": f"{result}",
        },
    ]
    return prompts


def generate_chat_prompt_baseline(task_instruction, input):
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
            "content": f"Input: {input}",
        },
        {
            "role": "user",
            "content": "Utilize your talent and critical thinking to provide a solution.",
        },
    ]
    return prompts
