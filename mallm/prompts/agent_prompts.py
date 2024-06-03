import json
import logging

from mallm.utils.types import TemplateFilling

logger = logging.getLogger("mallm")


def base_prompt(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {data.persona} ({data.persona_description}) \nYou are participating in a discussion. Answer in {data.sents_min} to {data.sents_max} sentences!",
        },
        {
            "role": "user",
            "content": f"Your Task: {data.task_instruction}\nInput: {data.input_str}\nPlease consider the task and input provided and think it step by step.",
        },
    ]
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "user",
                "content": f"This is the discussion to the current point.",
            }
        )
        prompts += data.agent_memory

    return prompts


def generate_chat_prompt_feedback(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    prompts.append(
        {
            "role": "user",
            "content": "Based on the current solution, give constructive feedback. Improve the answer considering the feedback.",
        }
    )
    return prompts


def generate_chat_prompt_agree(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    prompts.append(
        {
            "role": "user",
            "content": "Do you agree with the conclusion, considering the arguments and evidence presented? Please provide your reasoning step-by-step. After that respond with AGREE or DISAGREE.",
        }
    )
    return prompts


def generate_chat_prompt_improve(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory:
        prompts.append(
            {
                "role": "user",
                "content": "You dont agree with the current solution. Improve the current answer.",
            }
        )

    logger.debug(f"Sending prompt: {json.dumps(prompts, indent=2)}")

    return prompts


def generate_chat_prompt_draft(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    prompts.append(
        {
            "role": "user",
            "content": "Based on the provided feedback, carefully re-examine your previous solution and create a new draft.",
        }
    )

    logger.debug(f"Sending prompt: {json.dumps(prompts, indent=2)}")

    return prompts


def generate_final_answer_prompt(
    persona: str,
    persona_description: str,
    question: str,
    task: str,
    previous_answer: str,
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {persona} ({persona_description})",
        },
        {
            "role": "user",
            "content": f"You are tasked with creating a final answer based on the given question and your previous response.\nTask: {task}\nQuestion: {question}\nYour previous answer: {previous_answer}",
        },
        {
            "role": "user",
            "content": "Based on the above information, provide your final answer. Ensure your answer is comprehensive and well-considered.",
        },
    ]

    return prompts


def generate_voting_prompt(
    persona: str,
    persona_description: str,
    task: str,
    question: str,
    solutions: list[str],
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {persona} ({persona_description})",
        },
        {
            "role": "user",
            "content": f"You are tasked with voting for the best solution from the list provided below based on the given task.\nTask: {task}\nQuestion: {question}\n\nHere are the possible solutions:",
        },
    ]

    for i, solution in enumerate(solutions):
        prompts.append(
            {
                "role": "user",
                "content": f"Solution {i}: {solution}",
            }
        )

    prompts.append(
        {
            "role": "user",
            "content": "Based on the above solutions, please provide the number of the solution you are voting for. Answer only with the number.",
        }
    )

    return prompts
