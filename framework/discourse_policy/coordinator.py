import sys, os
sys.path.append('../../')
from framework.agents.agent import *

class Coordinator():
    def __init__(self, personas, task_name, task_description):
        self.personas = personas
        self.agents = []
        self.initAgents(task_name, task_description)
    

    def initAgents(self, task_name, task_description):
        for i, p in enumerate(self.personas):
            self.agents.append(Agent(i, "placeholder model", p))


Coordinator(["developer", "business manager", "ai researcher"], "Paraphrasing", "Please paraphrase... placeholder prompt")