def feedback():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {task_instruction}
Input: {input}
This is the recent feedback by others:
{agent_memory}
Here is the current solution you need to consider:
Solution: {inst_s}{current_draft}{inst_e}
Based on the current solution, give constructive feedback while considering your assigned role.
Your role: {persona} ({persona_description})
Utilize your talent and critical thinking to provide feedback in {sents_min}-{sents_max} sentences. If you agree, just answer with [AGREE]. If you [DISAGREE], explain why.{sys_e}\
'''
    return template

def improve():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {task_instruction}
Input: {input}
This is the recent feedback by others:
{agent_memory}
Here is the current solution you need to consider:
Solution: {inst_s}{current_draft}{inst_e}
Based on the current solution, carefully re-examine your previous answer while considering your assigned role.
Your role: {persona} ({persona_description})
Utilize your talent and critical thinking to provide feedback in {sents_min}-{sents_max} sentences and a new solution in as many sentences as you need. If you agree, just answer with [AGREE]. If you [DISAGREE], explain why.{sys_e}\
'''
    return template

def draft():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {task_instruction}
Input: {input}
This is the recent feedback by others:
{agent_memory}
Here is your last solution to the task which you need to reconsider:
Solution: {current_draft}
Based on the provided feedback, carefully re-examine your previous solution while considering your assigned role.
Your role: {persona} ({persona_description})
Utilize your talent and critical thinking to provide a new solution. If you feel like no changes are needed, output the existing solution.{sys_e}\
'''
    return template