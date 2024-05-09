from langchain_core.prompts import ChatPromptTemplate


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
                "content": "Based on the current solution, give constructive feedback. Be open to compromise too and if you agree answer with [AGREE] else answer with [DISAGREE] and explain why.",
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
                "content": "Improve the current answer and if you agree, answer with [AGREE] else answer with [DISAGREE] and explain why.",
            }
        )

    return prompts


def generate_chat_prompt_draft(data):
    # TODO improve this prompt when it is used
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
                "content": "Based on the provided feedback, carefully re-examine your previous solution. Be open to compromise too and if you agree, answer with [AGREE] else answer with [DISAGREE] and explain why.",
            }
        )

    return prompts
