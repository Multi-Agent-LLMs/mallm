import glob, os
from setup import *
import dbm
import json
import ast
import fire
from framework.prompts import agent_prompts
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

class Agent():
    def __init__(self, id, llm, persona, coordinator):
        self.id = id
        self.persona = persona
        self.memory_bucket = memory_bucket_dir+"agent_{}".format(self.id)
        self.coordinator = coordinator
        self.llm = llm

        self.chain_brainstorm = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.brainstorm()))
        self.chain_feedback = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.feedback()))
        self.chain_decide = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.decide_boolean()))
        self.chain_draft = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.draft()))

    def createDraft(self, unique_id, turn, task_instruction, input, agents_to_update, context_length=None):
        '''
        Initiates the creation of a draft by the agent. The draft takes preliminary discussion into account.
        Returns: string
        '''
        memory_string = self.getMemoryString(context_length)
        draft = self.chain_draft.invoke(
            {
                "task_instruction": task_instruction, 
                "persona": self.persona, 
                "input": input, 
                "agent_memory": memory_string
            })["text"]
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, "draft", draft)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "draft", draft)
        return draft
    
    def brainstorm(self, unique_id, turn, task_name, task_instruction, avg_feedback_length, agents_to_update, input, context_length=None):
        '''
        Initiates brainstorming for the agents with no preliminary discussion
        Returns: string
        '''
        res = self.chain_brainstorm.invoke(
            {
                "task_instruction": task_instruction, 
                "persona": self.persona, 
                "input": input
            })["text"]
        for a in agents_to_update:
            a.updateMemory(unique_id, turn, self.id, self.persona, "brainstorm", res)
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "brainstorm", res)
        return res

    def updateMemory(self, unique_id, turn, agent_id, agent_persona, contribution, text):
        '''
        Updates the dbm memory with another discussion entry.
        Returns string
        '''
        with dbm.open(self.memory_bucket, 'c') as db:
            db[str(unique_id)] = f'''{{"turn": {turn}, "agent_id": {agent_id}, "agent_persona": "{str(agent_persona).replace('"',"'")}", "contribution": "{contribution}", "text": "{str(text).replace('"',"'")}"}}'''
        self.saveMemoryToJson()
    
    def getMemory(self, context_length=None):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = {}
        with dbm.open(self.memory_bucket, 'r') as db:
            for key in db.keys():
                memory[key.decode()] = ast.literal_eval(db[key].decode().replace("\n", "\\n").replace("\t", "\\t"))
        if context_length:
            memory = memory.pop(range(0, len(memory)-context_length))
        return memory

    def getMemoryString(self, context_length=None, personalized = True):
        '''
        Retrieves memory from the agents memory bucket as a string
        Returns: string
        '''
        memory = self.getMemory(context_length)
        memory_string = ""
        for key in memory:
            if memory[key]["agent_persona"] != self.persona:
                memory_string = memory_string + f"\n[INST]{memory[key]["agent_persona"]}: {memory[key]["text"]}[/INST]"
            else:
                memory_string = memory_string + f"\n{memory[key]["agent_persona"]}: {memory[key]["text"]}"
        if personalized:
            memory_string = memory_string.replace(f"{self.persona}:", f"{self.persona} (you):")
        return memory_string
    
    def saveMemoryToJson(self, out=None):
        '''
        Converts the memory bucket dbm data to json format
        '''
        if out:
            try:
                with open(out, "w") as f: 
                    json.dump(self.getMemory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {out}: {e}")
        else:
            try:
                with open(self.memory_bucket+".json", "w") as f: 
                    json.dump(self.getMemory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {self.memory_bucket}: {e}")

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)