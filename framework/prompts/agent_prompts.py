def init_panelist(agent, num_agents, task_name, task_description):
    prompt = f"You are one of {num_agents} participants of a turn-based discourse with the goal to solve the ${task_name} task. {task_description}.
    This discussion format includes a moderator.
    Your role is: {persona}.
    In this discussion you are required to give constructive feedback about other drafts to solve the task that aligns with your beliefs. 
    Indicate whether you are satified with the current draft or whether changes should be made to solve the {task_name} task in a more optimal way."
    return prompt

def init_moderator(agent, task_name, task_description):
    prompt = f"You are one of {num_agents} participants of a turn-based discourse with the goal to solve the {task_name} task. {task_description}.
    You are the moderator in this discussion
    In this discussion you are required to be objective and non-opinionated. 
    The other participants will provide opinionated feedback on the current draft to solve the {task_name} task.
    You must try to incorporate everyones feedback as objectively as possible into a new draft that should satify everyones preferences to a maximal degree."
    return prompt