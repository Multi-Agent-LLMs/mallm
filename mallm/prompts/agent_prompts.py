from langchain_core.prompts import ChatPromptTemplate

feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "Please consider the example provided and think it step by step."),
        ("system", "Task: {taskInstruction}"),
        ("system", "Input: {input}"),
        ("system", "This is the recent feedback by others:\n{agentMemory}"),
        (
            "system",
            "Here is the current solution you need to consider:\nSolution: {currentDraft}",
        ),
        (
            "system",
            "Based on the current solution, give constructive feedback while considering your assigned role. Be "
            "open to compromise too.",
        ),
        ("system", "Your role: {persona} ({personaDescription})"),
        (
            "user",
            "Utilize your talent and critical thinking to provide feedback in {sentsMin}-{sentsMax} sentences. If "
            "you agree, answer with [AGREE] else with [DISAGREE]",
        ),
    ]
)


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


draft = ChatPromptTemplate.from_messages(
    [
        ("system", "Please consider the example provided and think it step by step."),
        ("system", "Task: {taskInstruction}"),
        ("system", "Input: {input}"),
        ("system", "This is the recent feedback by others:\n{agentMemory}"),
        (
            "system",
            "Here is your last solution to the task which you need to reconsider:\nSolution: {currentDraft}",
        ),
        (
            "system",
            "Based on the provided feedback, carefully re-examine your previous solution while considering your assigned "
            "role. Be open to compromise too.",
        ),
        ("system", "Your role: {persona} ({personaDescription})"),
        (
            "user",
            "Utilize your talent and critical thinking to provide a new solution. If you feel like no changes are needed, "
            "output the existing solution.",
        ),
    ]
)
