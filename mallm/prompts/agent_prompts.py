import json
import logging
from typing import Optional

from mallm.utils.types import TemplateFilling

logger = logging.getLogger("mallm")


def base_prompt(data: TemplateFilling) -> list[dict[str, str]]:
    appendix = ""
    if data.feedback_sentences is not None:
        appendix += f"\nExplain your reasoning in {data.feedback_sentences[0]} to {data.feedback_sentences[1]} sentences!"
    if data.current_draft is not None:
        appendix += f"\nHere is the current solution to the task: {data.current_draft}"
    else:
        appendix += f"\nNobody proposed a solution yet. Please provide the first one."

    if data.agent_memory is not None and data.agent_memory != []:
        appendix += f"\nThis is the discussion to the current point:"

    prompts = [
        {
            "role": "system",
            "content": f"You are participating in a discussion to solve the following task: {data.task_instruction} \nInput: {data.input_str} \nYour role: {data.persona} ({data.persona_description}) {appendix}",
        }
    ]
    if data.agent_memory is not None:
        prompts += data.agent_memory

    return prompts


def generate_chat_prompt_agree(data: TemplateFilling) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    prompts.append(
        {
            "role": "user",
            "content": "Do you agree with the conclusion, considering the arguments and evidence presented? Please provide your reasoning step-by-step. After that respond with [AGREE] or [DISAGREE].",
        }
    )
    return prompts


def generate_chat_prompt_feedback(
    data: TemplateFilling,
    chain_of_thought: bool,
    split_agree_and_answer: bool,
    agreement: Optional[bool],
) -> list[dict[str, str]]:
    prompts = base_prompt(data)

    if data.agent_memory:
        prefix = {
            None: "",
            True: "You agree with the current answer. ",
            False: "You disagree with the current answer. ",
        }[agreement]
        prompts.append(
            {
                "role": "user",
                "content": f"{prefix}Based on the current solution, give constructive feedback. Be open to compromise too.{'' if split_agree_and_answer else ' If you agree, answer with [AGREE], else answer with [DISAGREE] and explain why.'}",
            }
        )
        if chain_of_thought:
            prompts.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
                }
            )

    return prompts


def generate_chat_prompt_improve(
    data: TemplateFilling,
    chain_of_thought: bool,
    split_agree_and_answer: bool,
    agreement: Optional[bool],
) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory:
        prefix = {
            None: "",
            True: "You agree with the current answer. ",
            False: "You disagree with the current answer. ",
        }[agreement]
        prompts.append(
            {
                "role": "user",
                "content": f"{prefix}Improve the current answer.{'' if split_agree_and_answer else ' If you agree with the current answer, answer with [AGREE], else answer with [DISAGREE] and explain why!'}",
            }
        )
        if chain_of_thought:
            prompts.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
                }
            )

    logger.debug(f"Sending prompt: {json.dumps(prompts, indent=2)}")

    return prompts


def generate_chat_prompt_draft(
    data: TemplateFilling, chain_of_thought: bool
) -> list[dict[str, str]]:
    prompts = base_prompt(data)
    if data.agent_memory is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Based on the provided feedback, carefully re-examine your previous solution. Provide a revised solution based on the feedback.",
            }
        )
        if chain_of_thought:
            prompts.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
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


def generate_approval_voting_prompt(
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
            "content": f"You are tasked with approving any number of solutions from the list provided below based on the given task.\nTask: {task}\nQuestion: {question}\n\nHere are the possible solutions:",
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
            "content": "Based on the above solutions, please provide the numbers of the solutions you are approving, separated by commas. Answer only with the numbers.",
        }
    )

    return prompts


def generate_cumulative_voting_prompt(
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
            "content": f"You are tasked with distributing 10 points among the provided solutions based on the given task.\nTask: {task}\nQuestion: {question}\n\nHere are the possible solutions:",
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
            "content": "Based on the above solutions, please distribute 10 points among the solutions. Provide your points allocation as a JSON dictionary where keys are solution numbers (as int) and values are the points. The total points should sum up to 10. Answer only with the JSON dictionary.",
        }
    )

    return prompts


def generate_ranking_prompt(
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
            "content": f"You are tasked with ranking the solutions from the most preferred to the least preferred based on the given task.\nTask: {task}\nQuestion: {question}\n\nHere are the possible solutions:",
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
            "content": "Based on the above solutions, please provide the rankings of the solutions separated by spaces. Example: '0 2 1' if you prefer Solution 0 the most, then Solution 2, and finally Solution 1. Provide up to 5 rankings. Only answer with the rankings.",
        }
    )

    return prompts
