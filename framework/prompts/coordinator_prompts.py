def coordinate_personas(task_name, task_description, persona_type, source_text):
    prompt = f"The task to solve is {task_name}. {task_description} \
        The concrete example that needs to be solved is: '{source_text}' \
        You are the coordinator of a solution-oriented discussion plenum that consists of multiple {persona_type} participants. \
        Assess how many different participants would be optimal to solve the task and give a list of the {persona_type} roles involved in the discussion. \
        Give your output in this format: ```[A,B,C]```"
    return prompt