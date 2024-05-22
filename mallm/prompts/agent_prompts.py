import json
import logging

from mallm.utils.types import TemplateFilling

logger = logging.getLogger("mallm")


def base_prompt(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {data.persona} ({data.persona_description}) \nYou are participating in a discussion. Answer in {data.sents_min} to {data.sents_max} sentences!",
        }
    ]
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "system",
                "content": f"This is the discussion to the current point. Keep it in mind:\n{data.agent_memory}",
            }
        )

    prompts.append(
        {
            "role": "user",
            "content": f"Your Task: {data.task_instruction} Please consider the example provided and think it step by step. Input: {data.input_str}",
        }
    )

    if data.current_draft is not None:
        prompts.append(
            {
                "role": "user",
                "content": f"Here is the current solution you need to consider:\nSolution: {data.current_draft}",
            }
        )
    return prompts


def generate_chat_prompt_feedback(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Based on the current solution, give constructive feedback. Be open to compromise too. If you agree, answer with [AGREE], else answer with [DISAGREE] and explain why.",
            }
        )

    return prompts


def generate_chat_prompt_improve(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Improve the current answer and if you agree, answer with [AGREE] else answer with [DISAGREE] and repeat the answer.",
            }
        )

    logger.debug(f"Sending prompt: {json.dumps(prompts, indent=2)}")

    return prompts


def generate_chat_prompt_draft(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Based on the provided feedback, carefully re-examine your previous solution. Be open to compromise too and if you agree, answer with [AGREE] else answer with [DISAGREE] and explain why.",
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
