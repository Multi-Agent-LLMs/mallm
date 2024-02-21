from setup import *
import fire
from framework.prompts import agent_prompts
from framework.agents.agent import *

class Moderator(Agent):
    def __init__(self, id, llm, coordinator, persona = "Moderator"):
        self.id = id
        self.llm = llm
        self.persona = persona
        self.memory_bucket = memory_bucket_dir+"agent_{}".format(self.id)
        self.coordinator = coordinator

        self.chain_brainstorm = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.brainstorm()))
        self.chain_feedback = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.feedback()))
        self.chain_decide = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.decide_boolean()))
        self.chain_draft = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(agent_prompts.draft()))

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)