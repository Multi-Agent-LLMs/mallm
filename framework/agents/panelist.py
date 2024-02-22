from setup import *
import fire
from framework.prompts import agent_prompts
from framework.agents.agent import *

class Panelist(Agent):
    def generateFeedback(self, unique_id, turn, task_instruction, current_draft, agents_to_update, input, feedback_sentences, context_length=None):
        '''
        Initiates the generation of feedback relevant to the most recent draft. The feedback is generated based on the persona assigned to the agent. 
        The result is 1) the feedback and 2) a boolean vote if the agent agrees on the draft.
        Returns: string, boolean
        '''
        # generate the feedback output
        memory_string = self.getMemoryString(context_length)
        feedback = self.chain_feedback.invoke(
            {
                "task_instruction": task_instruction, 
                "persona": self.persona, 
                "agent_memory": memory_string, 
                "input": input, 
                "sentences": str(feedback_sentences)
            })["text"]
        # update the memory of this and other agents
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, "feedback", feedback)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "feedback", feedback)

        # extract the agreement boolean based on the generated feedback
        agreement = self.chain_decide.invoke(
            {
                "feedback": feedback
            })["text"]
        agreement_reason = agreement
        print(agreement)
        if "YES" in agreement and "NO" in agreement:
            agreement = None
        elif "NO" in agreement:
            agreement = False
        elif "YES" in agreement:
            agreement = True
        else:
            agreement = None
        print("->>> decision of this agent: " + str(agreement))

        return feedback, agreement, agreement_reason

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)