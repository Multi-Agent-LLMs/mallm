from setup import *
import fire
from framework.prompts import agent_prompts
from framework.agents.agent import *

class Moderator(Agent):
    def __init__(self, id, llm, coordinator):
        self.id = id
        self.llm = llm
        self.persona = "Moderator"
        self.memory_bucket = memory_bucket_dir+"agent_{}".format(self.id)
        self.coordinator = coordinator

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)