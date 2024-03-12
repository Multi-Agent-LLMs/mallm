from setup import *
import fire
from framework.agents.agent import *

class Panelist(Agent):
    
    def participate(self, use_moderator, memories, agreements, unique_id, turn, memory_ids, template_filling):
        '''
        Either calls feedback() or improve() depending on wether a moderator is present 
        '''
        if use_moderator:
            res, memory = self.feedback(
                unique_id = unique_id, 
                turn = turn, 
                memory_ids = memory_ids,
                template_filling = template_filling
                )
        else:
            res, memory = self.improve(
                unique_id = unique_id, 
                turn = turn, 
                memory_ids = memory_ids,
                template_filling = template_filling
                )
        memories.append(memory)
        memories = self.coordinator.updateMemories(memories, self.coordinator.agents)
        agreements = self.coordinator.agree(res, agreements)
        return memories, agreements

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)