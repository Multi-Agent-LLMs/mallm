import glob, os
from setup import *
import dbm
import json
import ast
import fire
from framework.prompts import agent_prompts

class Agent():
    def __init__(self, id, llm, persona, coordinator):
        self.id = id
        self.llm = llm
        self.persona = persona
        self.memory_bucket = memory_bucket_dir+"agent_{}".format(self.id)
        self.coordinator = coordinator

    def createDraft(self, unique_id, turn, task_name, task_description, source_text, agents_to_update, context_length=None, use_moderator=True):
        agent_memory = self.getMemory()
        if context_length:
            agent_memory = agent_memory.pop(range(0, len(agent_memory)-context_length))
        
        memory_string = ""
        for unique_id in agent_memory:
            memory_string = memory_string + f"\n Turn {agent_memory[key].turn}, {agent_memory[key].agent_persona}: {agent_memory[key].text}"

        draft = self.llm.invoke(agent_prompts.create_draft(task_name, task_description, agent_memory, self.persona, source_text, use_moderator))
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, draft)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, draft)
        print(draft)
        return draft

    def updateMemory(self, unique_id, turn, agent_id, agent_persona, text):
        with dbm.open(self.memory_bucket, 'c') as db:
            db[str(unique_id)] = f'{{"turn": {turn}, "agent_id": {agent_id}, "agent_persona": "{agent_persona}", "text": "{text}"}}'
        self.saveMemoryToJson()
    
    def getMemory(self):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = {}
        with dbm.open(self.memory_bucket, 'r') as db:
            for key in db.keys():
                memory[key.decode()] = ast.literal_eval(db[key].decode())
        print(type(memory))
        return memory
    
    def saveMemoryToJson(self):
        '''
        Converts the memory bucket dbm data to json format
        '''
        try:
            with open(self.memory_bucket+".json", "w") as f: 
                json.dump(self.getMemory(), f)
        except Exception as e:
            print(f"Failed to save agent memory to {self.memory_bucket}: {e}")

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)