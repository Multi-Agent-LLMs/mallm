import json
import logging
from typing import Optional

from mallm.agents.panelist import Panelist
from mallm.utils.types import TemplateFilling

logger = logging.getLogger("mallm")


def generate_chat_prompt_baseline(
    task_instruction: str, input_str: str, chain_of_thought: bool
) -> list[dict[str, str]]:
    # Use Zero-Shot-CoT: https://arxiv.org/pdf/2205.11916
    prompts = [
        {
            "role": "system",
            "content": f"Solve the following task: {task_instruction} \nInput: {input_str} \nProvide your solution in the end after writing 'FINAL SOLUTION:'.",
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


def base_prompt(data: TemplateFilling) -> list[dict[str, str]]:
    appendix = ""
    if data.current_draft is not None:
        appendix += f"\nHere is the current solution to the task: {data.current_draft}"
    else:
        appendix += "\nNobody proposed a solution yet. Please provide the first one."

    if data.agent_memory is not None and data.agent_memory != []:
        appendix += "\nThis is the discussion to the current point:"

    prompts = [
        {
            "role": "system",
            "content": f"You are participating in a discussion to solve the following task: {data.task_instruction} \nInput: {data.input_str} \nYour role: {data.persona} ({data.persona_description}) \nProvide your solution in the end after writing 'FINAL SOLUTION:'. {appendix}",
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
            "content": "Do you agree with the solution, considering the arguments and evidence presented? Please provide your reasoning step-by-step. After that respond with [AGREE] or [DISAGREE].",
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
            True: "You agree with the current solution. ",
            False: "You disagree with the current solution. ",
        }[agreement]
        prompts.append(
            {
                "role": "user",
                "content": f"{prefix}Improve the current solution.{'' if split_agree_and_answer else ' If you agree with the current solution, answer with [AGREE], else answer with [DISAGREE] and explain why and provide an improved solution!'}",
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
                "content": "Based on the provided feedback, carefully re-examine your previous solution. Provide a revised solution based on the feedback. Always declare a solution as 'FINAL SOLUTION:', even if it is just a starting point for further discussion.",
            }
        )
    else:
        prompts.append(
            {
                "role": "user",
                "content": "Provide a first solution. Declare this solution as 'FINAL SOLUTION:', even if it is just a starting point for further discussion.",
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
    return [
        {
            "role": "system",
            "content": f"Your role: {persona} ({persona_description})",
        },
        {
            "role": "user",
            "content": f"You are tasked with creating a final solution based on the given question and your previous response.\nTask: {task}\nQuestion: {question}\nYour previous solution: {previous_answer}",
        },
        {
            "role": "user",
            "content": "Extract the final solution to the task from the provided text. Remove statements of agreement, disagreement, and explanations. Do not modify the text. Do not output any text besides the solution.",
        },
    ]


def generate_answer_confidence_prompt(
    panelist: Panelist,
    question: str,
    task: str,
    final_answer: str,
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {panelist.persona} ({panelist.persona_description})",
        }
    ]
    discussion_history = panelist.get_discussion_history()[0]
    if discussion_history:
        prompts.append(
            {
                "role": "user",
                "content": "Here is the discussion history to help you make a decision:",
            }
        )
        prompts.extend(discussion_history)
    prompts.append(
        {
            "role": "user",
            "content": f"The task is: {task}. The question is: {question}. This is the final answer you provided: '{final_answer}'. Based on this information, please generate a confidence score between 0 and 100 for the final answer. Be critical abd only answer with the number.",
        }
    )
    return prompts


def voting_base_prompt(
    voting_message: str,
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {panelist.persona} ({panelist.persona_description})",
        }
    ]
    if history:
        discussion_history = panelist.get_discussion_history()[0]
        if discussion_history:
            prompts.append(
                {
                    "role": "user",
                    "content": "Here is the discussion history to help you make a decision:",
                }
            )
            prompts.extend(discussion_history)
    additional_context_str = (
        f"\nAdditional Context: {additional_context}" if additional_context else ""
    )
    content_str = (
        f"{voting_message}\n"
        f"Task: {task}\n"
        f"Question: {question}"
        f"{additional_context_str}\n\n"
        "Here are the possible solutions:"
    )
    prompts.append(
        {
            "role": "user",
            "content": content_str,
        }
    )
    for i, solution in enumerate(solutions):
        confidence_str = (
            "" if confidence is None else f"\n\n(Confidence: {round(confidence[i])} %)"
        )
        prompts.append(
            {
                "role": "user",
                "content": f"Solution {i if anonymous else panelists[i].persona}: {solution}{confidence_str}",
            }
        )
    return prompts


def generate_voting_prompt(
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = voting_base_prompt(
        "You are tasked with voting for the best solution from the list provided below based on the given task.",
        panelist,
        panelists,
        task,
        question,
        solutions,
        additional_context,
        anonymous,
        confidence,
        history,
    )

    prompts.append(
        {
            "role": "user",
            "content": "Based on the above solutions, please provide the number of the solution you are voting for. Answer only with the number.",
        }
    )

    return prompts


def generate_approval_voting_prompt(
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = voting_base_prompt(
        "You are tasked with approving any number of solutions from the list provided below based on the given task.",
        panelist,
        panelists,
        task,
        question,
        solutions,
        additional_context,
        anonymous,
        confidence,
        history,
    )

    prompts.append(
        {
            "role": "user",
            "content": "Based on the above solutions, please provide the numbers of the solutions you are approving, separated by commas. Answer only with the numbers.",
        }
    )

    return prompts


def generate_cumulative_voting_prompt(
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = voting_base_prompt(
        "You are tasked with distributing 10 points among the provided solutions based on the given task.",
        panelist,
        panelists,
        task,
        question,
        solutions,
        additional_context,
        anonymous,
        confidence,
        history,
    )

    prompts.append(
        {
            "role": "user",
            "content": "Based on the above solutions, please distribute 10 points among the solutions. Provide your points allocation as a JSON dictionary where keys are solution numbers (as int) and values are the points. The total points should sum up to 10. Answer only with the JSON dictionary.",
        }
    )

    return prompts


def generate_ranking_prompt(
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = voting_base_prompt(
        "You are tasked with ranking the solutions from the most preferred to the least preferred based on the given task.",
        panelist,
        panelists,
        task,
        question,
        solutions,
        additional_context,
        anonymous,
        confidence,
        history,
    )

    prompts.append(
        {
            "role": "user",
            "content": "Based on the above solutions, please provide the rankings of the solutions separated by spaces. Example: '0 2 1' if you prefer Solution 0 the most, then Solution 2, and finally Solution 1. Provide up to 5 rankings. Only answer with the rankings.",
        }
    )

    return prompts


def generate_summary_prompt(
    panelist: Panelist,
    panelists: list[Panelist],
    task: str,
    question: str,
    solutions: list[str],
    additional_context: Optional[str] = None,
    anonymous: bool = True,
    confidence: Optional[list[int]] = None,
    history: bool = False,
) -> list[dict[str, str]]:
    prompts = []

    # Add discussion history if available
    if history:
        discussion_history = panelist.get_discussion_history()[0]
        if discussion_history:
            prompts.append(
                {
                    "role": "user",
                    "content": "Here is the discussion history to help you make a decision:",
                }
            )
            prompts.extend(discussion_history)

    # Prepare the main content for the summary request
    additional_context_str = (
        f"\nAdditional Context: {additional_context}" if additional_context else ""
    )

    content_str = (
        f"Task: {task}\n"
        f"Question: {question}"
        f"{additional_context_str}\n\n"
        "Please provide a summary of the following solutions and combine them in a single answer to solve the task. Only answer with the solution:"
    )

    # Add each solution to the content string
    for i, solution in enumerate(solutions):
        confidence_str = (
            "" if confidence is None else f" (Confidence: {round(confidence[i])}%)"
        )
        panelist_label = f"Solution {i}" if anonymous else f"{panelists[i].persona}"
        content_str += f"\n\n{panelist_label}: {solution}{confidence_str}"

    # Append the final content as a user message
    prompts.append(
        {
            "role": "user",
            "content": content_str,
        }
    )

    # Return the prompts list
    return prompts
