import json
import logging

logger = logging.getLogger("mallm")


def generate_chat_prompt_feedback(data):
    # TODO improve this prompt when it is used
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {data['persona']} ({data['personaDescription']}) \nYou are participating in a discussion. Answer in {data['sentsMin']} to {data['sentsMax']} sentences!",
        }
    ]
    if data["agentMemory"] is not None:
        prompts.append(
            {
                "role": "system",
                "content": f"This is the discussion to the current point. Keep it in mind:\n{data['agentMemory']}",
            }
        )

    prompts.append(
        {
            "role": "user",
            "content": f"Your Task: {data['taskInstruction']} Please consider the example provided and think it step by step. Input: {data['input']}",
        }
    )

    if data["currentDraft"] is not None:
        prompts.append(
            {
                "role": "user",
                "content": f"Here is the current solution you need to consider:\nSolution: {data['currentDraft']}",
            }
        )

    if data["agentMemory"] is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Based on the current solution, give constructive feedback. Be open to compromise too. If you agree, answer with [AGREE], else answer with [DISAGREE] and explain why.",
            }
        )

    return prompts


def generate_chat_prompt_improve(data):
    prompts = [
        {
            "role": "system",
            "content": f"You are {data['persona']} and your traits are {data['personaDescription']} You are participating in a discussion. Answer in {data['sentsMin']} to {data['sentsMax']} sentences!",
        }
    ]
    if data["agentMemory"] is not None:
        prompts.append(
            {
                "role": "system",
                "content": "This is the discussion to the current point.",
            }
        )

        prompts += data["agentMemory"]

    prompts.append(
        {
            "role": "user",
            "content": f"Your Task: {data['taskInstruction']} Please consider the example provided and think it step by step. Input: {data['input']}",
        }
    )

    if data["currentDraft"] is not None:
        prompts.append(
            {
                "role": "user",
                "content": f"Here is the current solution you need to consider:\nSolution: {data['currentDraft']}",
            }
        )

    if data["agentMemory"] is not None:
        prompts.append(
            {
                "role": "user",
                "content": "Improve the current answer and if you agree, answer with [AGREE] else answer with [DISAGREE] and repeat the answer.",
            }
        )

    logger.debug(f"Sending prompt: {json.dumps(prompts, indent=2)}")

    return prompts


def generate_chat_prompt_draft(data):
    # TODO improve this prompt when it is used
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {data['persona']} ({data['personaDescription']}) \nYou are participating in a discussion. Propose a new solution based on the provided feedback.",
        }
    ]
    if data["agentMemory"] is not None:
        prompts.append(
            {
                "role": "system",
                "content": "This is the discussion to the current point.",
            }
        )
        prompts += data["agentMemory"]

    prompts.append(
        {
            "role": "user",
            "content": f"Your Task: {data['taskInstruction']} Please consider the example provided and think it step by step. Input: {data['input']}",
        }
    )

    if data["currentDraft"] is not None:
        prompts.append(
            {
                "role": "user",
                "content": f"Here is the current solution you need to consider:\nSolution: {data['currentDraft']}",
            }
        )

    if data["agentMemory"] is not None:
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
):
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {persona} ({persona_description}) \nYou are tasked with creating a final answer based on the given question and your previous response.",
        },
        {
            "role": "user",
            "content": f"Task: {task}\nQuestion: {question}\nYour previous answer: {previous_answer}",
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
):
    prompts = [
        {
            "role": "system",
            "content": f"Your role: {persona} ({persona_description}) \nYou are tasked with voting for the best solution from the list provided below based on the given task.",
        },
        {
            "role": "user",
            "content": f"Task: {task}\nQuestion: {question}\n\nHere are the possible solutions:",
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
