import glob, os
from setup import *
import dbm
import json
import fire

class Agent():
    def __init__(self, id, llm_model, persona):
        self.id = id
        self.llm = llm_model
        self.persona = persona
        self.memory_bucket = memory_bucket_dir+"agent_{}".format(self.id)
    
    def __del__(self):
        self.saveMemoryToJson()

    def createDraft():
        return None

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
                memory[key.decode()] = db[key].decode()
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

filelist = glob.glob(os.path.join(memory_bucket_dir, "*.bak"))
for f in filelist:
    os.remove(f)
filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dat"))
for f in filelist:
    os.remove(f)
filelist = glob.glob(os.path.join(memory_bucket_dir, "*.dir"))
for f in filelist:
    os.remove(f)
filelist = glob.glob(os.path.join(memory_bucket_dir, "*.json"))
for f in filelist:
    os.remove(f)

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)