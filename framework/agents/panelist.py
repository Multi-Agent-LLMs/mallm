from setup import *
import fire
from framework.prompts import agent_prompts
from framework.agents.agent import *

class Panelist(Agent):
    def generateFeedback(self, unique_id, turn, task_name, task_instruction, current_draft, avg_feedback_length, agents_to_update, source_text, context_length=None):
        '''
        Initiates the generation of feedback relevant to the most recent draft. The feedback is generated based on the persona assigned to the agent. 
        The result is 1) the feedback and 2) a boolean vote if the agent agrees on the draft.
        Returns: string, boolean
        '''
        # generate the feedback output
        memory_string = self.getMemoryString(context_length)
        feedback = self.chain_feedback.invoke({"task_instruction": task_instruction, "persona": self.persona, "agent_memory": memory_string, "source_text": source_text})["text"]
        # update the memory of this and other agents
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, feedback)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, feedback)

        # extract the agreement boolean based on the generated feedback
        agreement = self.chain_decide.invoke({"task_instruction": task_instruction, "current_draft": current_draft, "source_text": source_text, "feedback": feedback})["text"]
        print(agreement)
        agreement = "AGREE" in agreement
        print("->>> decision of this agent: " + str(agreement))

        return feedback, agreement

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)