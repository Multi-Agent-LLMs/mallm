def feedback():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {taskInstruction}
Input: {input}
This is the recent feedback by others:
{agentMemory}
Here is the current solution you need to consider:
Solution: {inst_s}{currentDraft}{inst_e}
Based on the current solution, give constructive feedback while considering your assigned role. Be open to compromise too.
Your role: {persona} ({personaDescription})
Utilize your talent and critical thinking to provide feedback in {sentsMin}-{sentsMax} sentences. If you agree, answer with [AGREE] and repeat the solution. If you [DISAGREE], explain why.{sys_e}\
'''
    return template


def improve():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {taskInstruction}
Input: {input}
This is the recent feedback by others:
{agentMemory}
Here is the current solution you need to consider:
Solution: {inst_s}{currentDraft}{inst_e}
Based on the current solution, carefully re-examine your previous answer while considering your assigned role. Be open to compromise too.
Your role: {persona} ({personaDescription})
Utilize your talent and critical thinking to provide feedback in {sentsMin}-{sentsMax} sentences and a new solution in as many sentences as you need. If you agree, answer with [AGREE] and repeat the solution. If you [DISAGREE], explain why.{sys_e}\
'''
    return template


def draft():
    template = '''\
{sys_s}Please consider the example provided and think it step by step.
Task: {taskInstruction}
Input: {input}
This is the recent feedback by others:
{agentMemory}
Here is your last solution to the task which you need to reconsider:
Solution: {currentDraft}
Based on the provided feedback, carefully re-examine your previous solution while considering your assigned role. Be open to compromise too.
Your role: {persona} ({personaDescription})
Utilize your talent and critical thinking to provide a new solution. If you feel like no changes are needed, output the existing solution.{sys_e}\
'''
    return template
