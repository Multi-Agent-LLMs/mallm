from setup import *
import fire
from framework.prompts import agent_prompts
from framework.agents.agent import *

class Panelist(Agent):
    def generateFeedback(self, unique_id, turn, task_name, task_description, current_draft, avg_feedback_length, agents_to_update, source_text, context_length=None):
        # generate the feedback output
        agent_memory = self.getMemory()
        if context_length:
            agent_memory = agent_memory.pop(range(0, len(agent_memory)-context_length))

        feedback = self.llm.invoke(agent_prompts.generate_feedback(task_name, task_description, agent_memory, current_draft, source_text, avg_feedback_length))
        # update the memory of this and other agents
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, feedback)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, feedback)
        print(feedback)
        
        # extract the agreement boolean based on the generated feedback
        agreement_text = self.llm.invoke(agent_prompts.decide_boolean(task_name, task_description, agent_memory, current_draft, feedback, source_text))
        print(agreement_text)
        agreement = "AGREE" in agreement_text
        print("decision: " + str(agreement))

        return feedback, agreement
    
    def brainstorm(self, unique_id, turn, task_name, task_description, avg_feedback_length, agents_to_update, source_text, context_length=None):
        brainstorm = self.llm.invoke(agent_prompts.brainstorm(task_name, task_description, source_text, avg_feedback_length))
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, brainstorm)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, brainstorm)
        print(brainstorm)
        return brainstorm

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)