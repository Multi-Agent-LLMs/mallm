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

improve = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are {persona} and your traits are {personaDescription}. Answer like you are this person! Here is the current solution you need to consider:\nSolution: {currentDraft}",
        ),
        ("system", "This is the recent feedback by others:\n{agentMemory}"),
        (
            "user",
            "Your Task: {taskInstruction}. Please consider the example provided and think it step by step. Input: {input}",
        ),
        (
            "user",
            "Improve the current answer and if you agree, answer with [AGREE] else answer with [DISAGREE] and explain why.",
        ),
    ]
)

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
