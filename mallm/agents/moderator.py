from mallm.agents.agent import *


class Moderator(Agent):
    def __init__(self, id, llm, coordinator, persona="Moderator"):
        self.id = id
        self.persona = persona
        self.persona_description = "A super-intelligent individual with critical thinking who has a neutral position at all times. He acts as a mediator between other discussion participants."
        self.memory_bucket = memory_bucket_dir + "agent_{}".format(self.id)
        self.coordinator = coordinator
        self.moderator = self
        self.llm = llm
        self.init_chains()


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
